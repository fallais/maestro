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
	if _, err := os.Stat(lib); err != nil {
		t.Skipf("missing %s (see README)", lib)
	}
	clip := loadF32(t, "clip")
	n := len(clip) / 2

	for _, tc := range []struct{ model, ref string }{
		{"htdemucs", "ref_stems"},
		{"htdemucs_6s", "ref_stems_htdemucs_6s"},
	} {
		t.Run(tc.model, func(t *testing.T) {
			modelPath := "../../models/" + tc.model + ".onnx"
			if _, err := os.Stat(modelPath); err != nil {
				t.Skipf("missing %s (tools/export_htdemucs.py %s)", modelPath, tc.model)
			}
			ref := loadF32(t, tc.ref)
			if err := InitRuntime(lib); err != nil {
				t.Fatal(err)
			}
			m, err := LoadONNX(modelPath, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Destroy()
			if len(ref) != len(m.Sources())*2*n {
				t.Fatalf("reference has %d stems, model reports %v", len(ref)/2/n, m.Sources())
			}

			start := time.Now()
			stems, err := Separate(context.Background(), m, clip[:n], clip[n:], nil)
			if err != nil {
				t.Fatal(err)
			}
			elapsed := time.Since(start).Seconds()
			t.Logf("%v: %.1f s of audio in %.1f s (%.2fx realtime)", m.Sources(), float64(n)/SampleRate, elapsed, float64(n)/SampleRate/elapsed)

			for s, name := range m.Sources() {
				for c := range 2 {
					db := snr(ref[(s*2+c)*n:(s*2+c+1)*n], stems[s][c])
					t.Logf("%s[%d] vs apply_model: SNR %.1f dB", name, c, db)
					if db < 40 {
						t.Errorf("%s[%d]: SNR %.1f dB < 40", name, c, db)
					}
				}
			}
		})
	}
}
