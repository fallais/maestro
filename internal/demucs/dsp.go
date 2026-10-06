// Package demucs runs htdemucs (exported by tools/export_htdemucs.py) with the
// STFT/iSTFT halves that cannot live in the ONNX graph implemented here.
package demucs

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/dsp/fourier"
)

// htdemucs constants. Must match tools/export_htdemucs.py.
const (
	SampleRate = 44100
	Segment    = 343980 // int(7.8 s * 44100), the training segment
	NFFT       = 4096
	Hop        = 1024
	Bins       = 2048 // NFFT / 2, Nyquist bin dropped by the model
	Frames     = 336  // ceil(Segment / Hop)
	Overlap    = 0.25 // demucs apply_model default

	MixSize     = 2 * Segment
	SpecSize    = 4 * Bins * Frames // [2 ch * (re, im), Bins, Frames]
	SpecOutSize = NumSources * SpecSize
	WaveOutSize = NumSources * MixSize
)

// NumSources is the number of stems htdemucs outputs.
const NumSources = 4

// Sources in model output order.
var Sources = [NumSources]string{"drums", "bass", "other", "vocals"}

// HTDemucs._spec pads the segment so FRAMES*HOP samples are covered, then
// torch.stft (center=True) reflect-pads NFFT/2 more per side and yields
// Frames+4 frames, of which _spec keeps [2, 2+Frames).
const (
	pad        = Hop / 2 * 3
	center     = NFFT / 2
	firstFrame = 2
	origin     = center + pad // segment sample 0 in torch.stft padded coordinates
	padRight   = pad + Frames*Hop - Segment
	innerLen   = Segment + pad + padRight
	plane      = Bins * Frames
)

var (
	scale       = 1 / math.Sqrt(NFFT) // normalized=True
	window      [NFFT]float64         // periodic Hann
	invEnvelope []float64             // 1 / Σ window² of all overlapping frames, per segment sample
)

func init() {
	for n := range window {
		window[n] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/NFFT)
	}
	env := make([]float64, Segment)
	for t := range Frames + 4 {
		for n := range NFFT {
			if s := t*Hop + n - origin; s >= 0 && s < Segment {
				env[s] += window[n] * window[n]
			}
		}
	}
	for i := range env {
		env[i] = 1 / env[i]
	}
	invEnvelope = env
}

// reflect mirrors i into [0, n) without repeating the edge (numpy "reflect").
// Pads are always shorter than n here, so one reflection suffices.
func reflect(i, n int) int {
	if i < 0 {
		return -i
	}
	if i >= n {
		return 2*(n-1) - i
	}
	return i
}

// Spectrogram computes the model's `spec` input from mix ([2, Segment],
// channel-major) into out ([4, Bins, Frames]: channel c's real part at plane 2c,
// imaginary part at plane 2c+1). Matches HTDemucs._spec + _magnitude.
func Spectrogram(mix, out []float32) {
	var wg sync.WaitGroup
	for c := range 2 {
		wg.Go(func() {
			x := mix[c*Segment : (c+1)*Segment]
			re := out[2*c*plane : (2*c+1)*plane]
			im := out[(2*c+1)*plane : (2*c+2)*plane]
			fft := fourier.NewFFT(NFFT)
			frame := make([]float64, NFFT)
			coeffs := make([]complex128, NFFT/2+1)
			for t := range Frames {
				start := (t + firstFrame) * Hop
				for n := range NFFT {
					// Two nested reflect pads: torch.stft's, then _spec's.
					inner := reflect(start+n-center, innerLen)
					frame[n] = float64(x[reflect(inner-pad, Segment)]) * window[n]
				}
				fft.Coefficients(coeffs, frame)
				for k := range Bins {
					re[k*Frames+t] = float32(real(coeffs[k]) * scale)
					im[k*Frames+t] = float32(imag(coeffs[k]) * scale)
				}
			}
		})
	}
	wg.Wait()
}

// AddInverseSpectrogram turns the model outputs into stem waveforms in place:
// wave += iSTFT(specOut). specOut is [Sources, 4, Bins, Frames], wave is
// [Sources, 2, Segment]. Matches HTDemucs._mask + _ispec.
func AddInverseSpectrogram(specOut, wave []float32) {
	var wg sync.WaitGroup
	for sc := range NumSources * 2 {
		wg.Go(func() {
			re := specOut[2*sc*plane : (2*sc+1)*plane]
			im := specOut[(2*sc+1)*plane : (2*sc+2)*plane]
			fft := fourier.NewFFT(NFFT)
			coeffs := make([]complex128, NFFT/2+1) // Nyquist bin stays 0, as _ispec pads it
			frame := make([]float64, NFFT)
			ola := make([]float64, Segment)
			// Sequence is unnormalized (×NFFT); undo the forward's `scale` too.
			gain := 1 / (scale * NFFT)
			// _ispec zero-pads 2 frames per side; only the real Frames carry signal.
			for t := range Frames {
				for k := range Bins {
					coeffs[k] = complex(float64(re[k*Frames+t]), float64(im[k*Frames+t]))
				}
				fft.Sequence(frame, coeffs)
				offset := (t+firstFrame)*Hop - origin
				n0, n1 := max(0, -offset), min(NFFT, Segment-offset)
				for n := n0; n < n1; n++ {
					ola[offset+n] += frame[n] * gain * window[n]
				}
			}
			out := wave[sc*Segment : (sc+1)*Segment]
			for s := range out {
				out[s] += float32(ola[s] * invEnvelope[s])
			}
		})
	}
	wg.Wait()
}
