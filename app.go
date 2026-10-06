package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	wruntime "github.com/wailsapp/wails/v2/pkg/runtime"

	"maestro/internal/audio"
	"maestro/internal/demucs"
)

// Peak RSS of the htdemucs CPU session (weights + activations, arena off), measured.
const modelMemory = 3 << 30

// App is bound to the frontend; its exported methods become JS functions.
type App struct {
	ctx  context.Context
	emit func(event string, data any) // wruntime.EventsEmit; replaced in tests

	mu      sync.Mutex
	model   *demucs.ONNXModel
	track   *Track
	left    []float32 // decoded mix of the current track (44.1 kHz)
	right   []float32
	cancel  context.CancelFunc
	running bool

	lastDir     string // folder of the last opened song, for the file dialog
	initialPath string // file passed on the command line, opened once
	selfTest    bool   // --selftest: frontend plays briefly, logs and quits
}

// Track describes the loaded song.
type Track struct {
	ID       string  `json:"id"`
	Name     string  `json:"name"`
	Duration float64 `json:"duration"`
	MixURL   string  `json:"mixUrl"`
	// Stems is set once separated (possibly from the cache).
	Stems []Stem `json:"stems"`
	dir   string
}

// Stem is one separated source, playable at URL.
type Stem struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}

// SeparationProgress is emitted as the "separation:progress" event.
type SeparationProgress struct {
	Stage     string  `json:"stage"` // "loading-model" | "separating" | "writing"
	Segment   int     `json:"segment"`
	Segments  int     `json:"segments"`
	ElapsedMs float64 `json:"elapsedMs"`
}

func NewApp(initialPath string, selfTest bool) *App {
	return &App{initialPath: initialPath, selfTest: selfTest}
}

// SelfTest reports whether the app was started with --selftest.
func (a *App) SelfTest() bool { return a.selfTest }

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	a.emit = func(event string, data any) { wruntime.EventsEmit(ctx, event, data) }
}

func (a *App) shutdown(context.Context) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.cancel != nil {
		a.cancel()
	}
	if a.model != nil {
		a.model.Destroy()
	}
}

// InitialTrack loads the file given on the command line, if any (only once).
func (a *App) InitialTrack() (*Track, error) {
	a.mu.Lock()
	path := a.initialPath
	a.initialPath = ""
	a.mu.Unlock()
	if path == "" {
		return nil, nil
	}
	return a.LoadTrack(path)
}

// LogFrontend writes a frontend message to the app log (release builds have no devtools).
func (a *App) LogFrontend(level, msg string) {
	log.Printf("[frontend %s] %s", level, msg)
}

var audioExts = []string{"mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "webm"}

// OpenAudio shows a file dialog and loads the chosen file. Returns nil if cancelled.
func (a *App) OpenAudio() (*Track, error) {
	// GTK patterns are case-sensitive: match SONG.MP3 too.
	var patterns []string
	for _, ext := range audioExts {
		patterns = append(patterns, "*."+ext, "*."+strings.ToUpper(ext))
	}
	path, err := wruntime.OpenFileDialog(a.ctx, wruntime.OpenDialogOptions{
		Title: "Open a song",
		// Without a folder, GTK opens on "Recent", which hides freshly downloaded files.
		DefaultDirectory: a.dialogDir(),
		Filters: []wruntime.FileFilter{
			{DisplayName: "Audio files", Pattern: strings.Join(patterns, ";")},
			{DisplayName: "All files", Pattern: "*"},
		},
	})
	if err != nil || path == "" {
		return nil, err
	}
	return a.LoadTrack(path)
}

// LoadTrack decodes a file to 44.1 kHz stereo and prepares its cache directory.
func (a *App) LoadTrack(path string) (*Track, error) {
	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		return nil, errors.New("a separation is running; cancel it first")
	}
	a.mu.Unlock()

	id, err := fileID(path)
	if err != nil {
		return nil, err
	}
	root, err := dataDir()
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(root, "tracks", id)
	if err := os.MkdirAll(filepath.Join(dir, "stems"), 0o755); err != nil {
		return nil, err
	}

	left, right, err := audio.DecodeStereo(a.ctx, path, demucs.SampleRate)
	if err != nil {
		return nil, err
	}
	mixPath := filepath.Join(dir, "mix.wav")
	if _, err := os.Stat(mixPath); err != nil {
		if err := audio.WriteWAV16(mixPath, left, right, demucs.SampleRate); err != nil {
			return nil, fmt.Errorf("writing mix: %w", err)
		}
	}

	t := &Track{
		ID:       id,
		Name:     strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		Duration: float64(len(left)) / demucs.SampleRate,
		MixURL:   mediaURL(id, "mix.wav"),
		dir:      dir,
	}
	t.Stems = cachedStems(t)

	a.mu.Lock()
	a.track, a.left, a.right = t, left, right
	a.lastDir = filepath.Dir(path)
	a.mu.Unlock()
	return t, nil
}

// SeparateStems runs Demucs on the loaded track (or returns cached stems),
// emitting "separation:progress" events.
func (a *App) SeparateStems() ([]Stem, error) {
	a.mu.Lock()
	t, left, right := a.track, a.left, a.right
	switch {
	case t == nil:
		a.mu.Unlock()
		return nil, errNoTrack
	case len(t.Stems) > 0:
		a.mu.Unlock()
		return t.Stems, nil
	case a.running:
		a.mu.Unlock()
		return nil, errors.New("a separation is already running")
	}
	ctx, cancel := context.WithCancel(a.ctx)
	a.cancel, a.running = cancel, true
	a.mu.Unlock()
	defer func() {
		cancel()
		a.mu.Lock()
		a.cancel, a.running = nil, false
		a.mu.Unlock()
	}()

	// Go cannot recover from running out of memory, so refuse up front.
	need := uint64(len(left))*4*(2+2*demucs.NumSources+1) + modelMemory
	if avail, ok := availableMemory(); ok && need > avail {
		return nil, fmt.Errorf("not enough memory: separating %.0f s of audio needs about %.1f GB, %.1f GB available. Close other apps or use a shorter file",
			t.Duration, float64(need)/(1<<30), float64(avail)/(1<<30))
	}

	start := time.Now()
	emit := func(p SeparationProgress) {
		p.ElapsedMs = float64(time.Since(start).Milliseconds())
		a.emit("separation:progress", p)
	}

	emit(SeparationProgress{Stage: "loading-model"})
	model, err := a.loadModel()
	if err != nil {
		return nil, err
	}

	segments := demucs.SegmentCount(len(left))
	emit(SeparationProgress{Stage: "separating", Segments: segments})
	stems, err := demucs.Separate(ctx, model, left, right, func(s, n int) {
		emit(SeparationProgress{Stage: "separating", Segment: s, Segments: n})
	})
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return nil, errors.New("separation cancelled")
		}
		return nil, err
	}

	emit(SeparationProgress{Stage: "writing", Segment: segments, Segments: segments})
	var wg sync.WaitGroup
	errs := make([]error, demucs.NumSources)
	for s, name := range demucs.Sources {
		wg.Go(func() {
			// Write to a temp name so an interrupted run never looks cached.
			final := filepath.Join(t.dir, "stems", name+".wav")
			errs[s] = audio.WriteWAV16(final+".tmp", stems[s][0], stems[s][1], demucs.SampleRate)
			if errs[s] == nil {
				errs[s] = os.Rename(final+".tmp", final)
			}
		})
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return nil, fmt.Errorf("writing stems: %w", err)
	}

	a.mu.Lock()
	t.Stems = cachedStems(t)
	a.mu.Unlock()
	return t.Stems, nil
}

// CancelSeparation stops a running separation after the current segment.
func (a *App) CancelSeparation() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.cancel != nil {
		a.cancel()
	}
}

// ExportStem asks where to save a stem (or "mix") and copies its WAV there.
func (a *App) ExportStem(name string) (string, error) {
	a.mu.Lock()
	t := a.track
	a.mu.Unlock()
	if t == nil {
		return "", errNoTrack
	}
	src := filepath.Join(t.dir, "stems", name+".wav")
	if name == "mix" {
		src = filepath.Join(t.dir, "mix.wav")
	} else if !slices.Contains(demucs.Sources[:], name) {
		return "", fmt.Errorf("unknown stem %q", name)
	}
	dst, err := wruntime.SaveFileDialog(a.ctx, wruntime.SaveDialogOptions{
		Title:           "Export " + name,
		DefaultFilename: fmt.Sprintf("%s - %s.wav", t.Name, name),
		Filters:         []wruntime.FileFilter{{DisplayName: "WAV", Pattern: "*.wav"}},
	})
	if err != nil || dst == "" {
		return "", err
	}
	return dst, copyFile(src, dst)
}

// dialogDir is the last song's folder, else ~/Music, else home (it must exist).
func (a *App) dialogDir() string {
	a.mu.Lock()
	last := a.lastDir
	a.mu.Unlock()
	home, _ := os.UserHomeDir()
	for _, d := range []string{last, filepath.Join(home, "Music"), home} {
		if st, err := os.Stat(d); d != "" && err == nil && st.IsDir() {
			return d
		}
	}
	return ""
}

func (a *App) loadModel() (*demucs.ONNXModel, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.model != nil {
		return a.model, nil
	}
	lib, err := findORTLib()
	if err != nil {
		return nil, err
	}
	if err := demucs.InitRuntime(lib); err != nil {
		return nil, err
	}
	path, err := findModel("htdemucs")
	if err != nil {
		return nil, err
	}
	m, err := demucs.LoadONNX(path, runtime.NumCPU())
	if err != nil {
		return nil, err
	}
	a.model = m
	return m, nil
}

func cachedStems(t *Track) []Stem {
	var stems []Stem
	for _, name := range demucs.Sources {
		if _, err := os.Stat(filepath.Join(t.dir, "stems", name+".wav")); err != nil {
			return nil
		}
		stems = append(stems, Stem{Name: name, URL: mediaURL(t.ID, "stems/"+name+".wav")})
	}
	return stems
}

// fileID hashes the file content, so renamed/moved songs reuse their cache.
func fileID(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil))[:16], nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func mediaURL(id, file string) string { return "/media/" + id + "/" + file }

var mediaPath = regexp.MustCompile(`^/media/([0-9a-f]{16})/(mix\.wav|stems/(drums|bass|other|vocals)\.wav)$`)

// mediaHandler serves cached WAVs to the webview (with Range support for seeking).
func mediaHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m := mediaPath.FindStringSubmatch(r.URL.Path)
		root, err := dataDir()
		if m == nil || err != nil {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(root, "tracks", m[1], filepath.FromSlash(m[2])))
	})
}
