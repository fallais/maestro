package demucs

import (
	"context"
	"errors"
	"math"
)

// Model runs the ONNX graph on one segment: mix [2, Segment] and spec
// [4, Bins, Frames] in; specOut [S, 4, Bins, Frames] and waveOut [S, 2, Segment]
// out, for the S = len(Sources()) sources. The returned slices are only valid
// until the next call; waveOut may be modified by the caller. Cancelling ctx
// should abort an in-flight run.
type Model interface {
	Sources() []string
	Run(ctx context.Context, mix, spec []float32) (specOut, waveOut []float32, err error)
}

// Stems holds stem waveforms indexed [source][channel], in Model.Sources() order.
type Stems [][2][]float32

// Progress reports completed segments.
type Progress func(segment, segments int)

func stride() int { return int((1 - Overlap) * Segment) }

// SegmentCount is the number of model calls Separate makes for length samples.
func SegmentCount(length int) int { return (length + stride() - 1) / stride() }

// Separate splits a 44.1 kHz stereo track into stems. It ports demucs
// apply_model(shifts=0, split=True, overlap=0.25) plus the whole-track
// normalization of demucs.api.Separator.
func Separate(ctx context.Context, m Model, left, right []float32, progress Progress) (Stems, error) {
	length := len(left)
	if len(right) != length {
		return nil, errors.New("channel length mismatch")
	}
	if length == 0 {
		return nil, errors.New("empty audio")
	}

	// Normalize the whole track by the mono mix statistics (unbiased std).
	var sum float64
	for i := range left {
		sum += float64(left[i]+right[i]) / 2
	}
	mean := sum / float64(length)
	var sq float64
	for i := range left {
		d := float64(left[i]+right[i])/2 - mean
		sq += d * d
	}
	std := math.Sqrt(sq/float64(max(1, length-1))) + 1e-8

	stems := make(Stems, len(m.Sources()))
	for s := range stems {
		stems[s] = [2][]float32{make([]float32, length), make([]float32, length)}
	}
	weightSum := make([]float32, length)

	// Triangular weights peaking mid-segment.
	weight := make([]float32, Segment)
	half := Segment / 2
	for i := range weight {
		if i < half {
			weight[i] = float32(i+1) / float32(half)
		} else {
			weight[i] = float32(Segment-i) / float32(half)
		}
	}

	mix := make([]float32, MixSize)
	spec := make([]float32, SpecSize)
	segments := SegmentCount(length)
	in := [2][]float32{left, right}

	for index, offset := 0, 0; offset < length; index, offset = index+1, offset+stride() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		chunk := min(Segment, length-offset)
		// TensorChunk.padded: center the chunk in a full segment, using real audio
		// where it exists and zeros past the track edges.
		start := offset - (Segment-chunk)/2
		clear(mix)
		for c := range 2 {
			for i := max(0, start); i < min(length, start+Segment); i++ {
				mix[c*Segment+i-start] = float32((float64(in[c][i]) - mean) / std)
			}
		}

		Spectrogram(mix, spec)
		specOut, waveOut, err := m.Run(ctx, mix, spec)
		if err != nil {
			return nil, err
		}
		AddInverseSpectrogram(specOut, waveOut)

		// center_trim back to the chunk, then weighted overlap-add.
		trim := (Segment - chunk) / 2
		for s := range stems {
			for c := range 2 {
				src := waveOut[(s*2+c)*Segment+trim:]
				dst := stems[s][c][offset:]
				for i := range chunk {
					dst[i] += src[i] * weight[i]
				}
			}
		}
		for i := range chunk {
			weightSum[offset+i] += weight[i]
		}
		if progress != nil {
			progress(index+1, segments)
		}
	}

	for s := range stems {
		for c := range 2 {
			ch := stems[s][c]
			for i := range ch {
				ch[i] = float32(float64(ch[i]/weightSum[i])*std + mean)
			}
		}
	}
	return stems, nil
}
