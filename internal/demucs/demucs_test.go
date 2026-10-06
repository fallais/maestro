package demucs

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Fixtures come from tools/make_reference.py (PyTorch demucs dumps).
const fixtures = "../../tools/fixtures"

func loadF32(t *testing.T, name string) []float32 {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(fixtures, name+".f32"))
	if os.IsNotExist(err) {
		t.Skipf("missing fixture %s: run tools/make_reference.py", name)
	}
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// snr is the signal-to-difference ratio of est against ref, in dB.
func snr(ref, est []float32) float64 {
	var s, e float64
	for i := range ref {
		s += float64(ref[i]) * float64(ref[i])
		d := float64(ref[i] - est[i])
		e += d * d
	}
	return 10 * math.Log10(s/math.Max(e, 1e-30))
}

func TestSpectrogramMatchesTorch(t *testing.T) {
	spec := make([]float32, SpecSize)
	Spectrogram(loadF32(t, "seg_mix"), spec)
	if db := snr(loadF32(t, "seg_spec"), spec); db < 100 {
		t.Fatalf("STFT vs torch _spec: SNR %.1f dB", db)
	} else {
		t.Logf("STFT vs torch _spec: SNR %.1f dB", db)
	}
}

func TestInverseSpectrogramMatchesTorch(t *testing.T) {
	wave := loadF32(t, "seg_wave_out")
	AddInverseSpectrogram(loadF32(t, "seg_spec_out"), wave)
	if db := snr(loadF32(t, "seg_full"), wave); db < 100 {
		t.Fatalf("iSTFT vs torch _ispec: SNR %.1f dB", db)
	} else {
		t.Logf("iSTFT vs torch _ispec: SNR %.1f dB", db)
	}
}

func TestSeparateMatchesPyTorch(t *testing.T) {
	lib := os.Getenv("MAESTRO_ORT_LIB")
	if lib == "" {
		lib = "../../third_party/onnxruntime/onnxruntime-linux-x64-1.29.1/lib/libonnxruntime.so.1.29.1"
	}
	modelPath := os.Getenv("MAESTRO_MODEL")
	if modelPath == "" {
		modelPath = "../../models/htdemucs.onnx"
	}
	for _, p := range []string{lib, modelPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s (see README)", p)
		}
	}
	clip := loadF32(t, "clip")
	ref := loadF32(t, "ref_stems")

	if err := InitRuntime(lib); err != nil {
		t.Fatal(err)
	}
	m, err := LoadONNX(modelPath, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Destroy()

	n := len(clip) / 2
	start := time.Now()
	stems, err := Separate(context.Background(), m, clip[:n], clip[n:], func(s, total int) {
		t.Logf("segment %d/%d at %.1fs", s, total, time.Since(start).Seconds())
	})
	if err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start).Seconds()
	t.Logf("%.1f s of audio in %.1f s (%.2fx realtime)", float64(n)/SampleRate, elapsed, float64(n)/SampleRate/elapsed)

	for s, name := range Sources {
		for c := range 2 {
			db := snr(ref[(s*2+c)*n:(s*2+c+1)*n], stems[s][c])
			t.Logf("%s[%d] vs apply_model: SNR %.1f dB", name, c, db)
			if db < 40 {
				t.Errorf("%s[%d]: SNR %.1f dB < 40", name, c, db)
			}
		}
	}
}
