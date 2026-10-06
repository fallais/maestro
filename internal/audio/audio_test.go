package audio

import (
	"context"
	"math"
	"os/exec"
	"path/filepath"
	"testing"
)

// Round trip: WriteWAV16 at 48 kHz, then DecodeStereo resamples it to 44.1 kHz.
func TestWriteThenDecodeResamples(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not installed")
	}
	const inRate, outRate, seconds = 48000, 44100, 2
	left := make([]float32, inRate*seconds)
	right := make([]float32, inRate*seconds)
	for i := range left {
		left[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/inRate))
		right[i] = float32(0.25 * math.Sin(2*math.Pi*220*float64(i)/inRate))
	}
	path := filepath.Join(t.TempDir(), "tone.wav")
	if err := WriteWAV16(path, left, right, inRate); err != nil {
		t.Fatal(err)
	}

	l, r, err := DecodeStereo(context.Background(), path, outRate)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(l), outRate*seconds; abs(got-want) > 64 {
		t.Fatalf("length %d, want ~%d", got, want)
	}
	// Compare against the ideal 44.1 kHz signal away from the edges.
	var errL, errR float64
	for i := 1000; i < len(l)-1000; i++ {
		errL = math.Max(errL, math.Abs(float64(l[i])-0.5*math.Sin(2*math.Pi*440*float64(i)/outRate)))
		errR = math.Max(errR, math.Abs(float64(r[i])-0.25*math.Sin(2*math.Pi*220*float64(i)/outRate)))
	}
	if errL > 1e-3 || errR > 1e-3 {
		t.Fatalf("resampled signal off: max err L %.2e R %.2e", errL, errR)
	}
}

func abs(x int) int { return max(x, -x) }
