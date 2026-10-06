package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// newTestApp returns an App outside Wails, with events recorded and the
// per-song cache in a temp dir. Skips if the model or ORT library is missing.
func newTestApp(t *testing.T) (*App, func() []SeparationProgress) {
	t.Helper()
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	if _, err := findORTLib(); err != nil {
		t.Skip(err)
	}
	if _, err := findModel("htdemucs"); err != nil {
		t.Skip(err)
	}
	if _, err := os.Stat("tools/fixtures/clip.wav"); err != nil {
		t.Skip("run tools/make_reference.py first")
	}
	var mu sync.Mutex
	var events []SeparationProgress
	a := &App{ctx: context.Background(), emit: func(name string, data any) {
		mu.Lock()
		defer mu.Unlock()
		if p, ok := data.(SeparationProgress); ok && name == "separation:progress" {
			events = append(events, p)
		}
	}}
	t.Cleanup(func() { a.shutdown(context.Background()) })
	return a, func() []SeparationProgress {
		mu.Lock()
		defer mu.Unlock()
		return append([]SeparationProgress(nil), events...)
	}
}

func TestLoadSeparateServeAndCache(t *testing.T) {
	a, events := newTestApp(t)

	track, err := a.LoadTrack("tools/fixtures/clip.wav")
	if err != nil {
		t.Fatal(err)
	}
	if track.Duration < 11.9 || track.Duration > 12.1 || len(track.Stems) != 0 {
		t.Fatalf("unexpected track %+v", track)
	}

	start := time.Now()
	stems, err := a.SeparateStems()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("separated in %.1fs", time.Since(start).Seconds())
	if len(stems) != 4 {
		t.Fatalf("got %d stems", len(stems))
	}
	var stages []string
	for _, e := range events() {
		if len(stages) == 0 || stages[len(stages)-1] != e.Stage {
			stages = append(stages, e.Stage)
		}
	}
	if got := strings.Join(stages, ","); got != "loading-model,separating,writing" {
		t.Errorf("stages %q", got)
	}

	// The media handler serves the stems with Range support (needed for seeking).
	srv := httptest.NewServer(mediaHandler())
	defer srv.Close()
	req, _ := http.NewRequest("GET", srv.URL+stems[1].URL, nil)
	req.Header.Set("Range", "bytes=0-43")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusPartialContent {
		t.Errorf("range request: %s", res.Status)
	}
	for _, bad := range []string{"/media/../../etc/passwd", "/media/0123456789abcdef/stems/x.wav"} {
		res, err := http.Get(srv.URL + bad)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		if res.StatusCode != http.StatusNotFound {
			t.Errorf("%s: %s, want 404", bad, res.Status)
		}
	}

	// Reloading the same file hits the stem cache.
	again, err := a.LoadTrack("tools/fixtures/clip.wav")
	if err != nil {
		t.Fatal(err)
	}
	if len(again.Stems) != 4 {
		t.Errorf("expected cached stems on reload, got %d", len(again.Stems))
	}
}

func TestCancelSeparation(t *testing.T) {
	a, events := newTestApp(t)
	if _, err := a.LoadTrack("tools/fixtures/clip.wav"); err != nil {
		t.Fatal(err)
	}
	var cancelAt time.Time
	go func() {
		for len(events()) == 0 || events()[len(events())-1].Segment < 1 {
			time.Sleep(10 * time.Millisecond)
		}
		time.Sleep(500 * time.Millisecond) // mid-way through segment 2
		cancelAt = time.Now()
		a.CancelSeparation()
	}()
	if _, err := a.SeparateStems(); err == nil || !strings.Contains(err.Error(), "cancelled") {
		t.Fatalf("want cancellation error, got %v", err)
	}
	latency := time.Since(cancelAt)
	t.Logf("cancel latency %v", latency.Round(time.Millisecond))
	if latency > time.Second {
		t.Errorf("cancel took %v; the in-flight segment should be terminated", latency)
	}
	if stems := cachedStems(a.track); stems != nil {
		t.Error("a cancelled run must not leave cached stems")
	}
}
