// Package audio decodes input files with ffmpeg and writes WAV files.
package audio

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"strconv"
)

// ErrNoFFmpeg means ffmpeg is not installed or not on PATH.
var ErrNoFFmpeg = errors.New("ffmpeg not found on PATH: install it (e.g. `sudo apt install ffmpeg`) to open audio files")

// DecodeStereo decodes any ffmpeg-supported file (mp3, wav, m4a, flac, ...) to
// stereo float32 at sampleRate. Mono is duplicated; more channels are downmixed.
func DecodeStereo(ctx context.Context, path string, sampleRate int) (left, right []float32, err error) {
	bin, err := exec.LookPath("ffmpeg")
	if err != nil {
		return nil, nil, ErrNoFFmpeg
	}
	cmd := exec.CommandContext(ctx, bin, "-nostdin", "-v", "error", "-i", path,
		"-vn", "-ac", "2", "-ar", strconv.Itoa(sampleRate),
		"-af", "aresample=resampler=soxr", "-f", "f32le", "-")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, err
	}

	r := bufio.NewReaderSize(stdout, 1<<20)
	var frame [8]byte
	for {
		if _, err := io.ReadFull(r, frame[:]); err != nil {
			break // EOF (or a short final frame)
		}
		left = append(left, math.Float32frombits(binary.LittleEndian.Uint32(frame[0:])))
		right = append(right, math.Float32frombits(binary.LittleEndian.Uint32(frame[4:])))
	}
	if err := cmd.Wait(); err != nil {
		if ctx.Err() != nil {
			return nil, nil, ctx.Err()
		}
		return nil, nil, fmt.Errorf("could not decode %s: %s", path, bytes.TrimSpace(stderr.Bytes()))
	}
	if len(left) == 0 {
		return nil, nil, fmt.Errorf("%s contains no audio", path)
	}
	return left, right, nil
}

// WriteWAV16 writes stereo samples as a 16-bit PCM WAV with TPDF dither.
func WriteWAV16(path string, left, right []float32, sampleRate int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	dataSize := uint32(len(left) * 4)
	header := []any{
		[4]byte{'R', 'I', 'F', 'F'}, 36 + dataSize, [4]byte{'W', 'A', 'V', 'E'},
		[4]byte{'f', 'm', 't', ' '}, uint32(16), uint16(1), uint16(2),
		uint32(sampleRate), uint32(sampleRate * 4), uint16(4), uint16(16),
		[4]byte{'d', 'a', 't', 'a'}, dataSize,
	}
	for _, v := range header {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			f.Close()
			return err
		}
	}
	var rng uint32 = 0x9e3779b9
	dither := func() float64 { // triangular, ±1 LSB
		rng ^= rng << 13
		rng ^= rng >> 17
		rng ^= rng << 5
		a := float64(rng) / math.MaxUint32
		rng ^= rng << 13
		rng ^= rng >> 17
		rng ^= rng << 5
		return a - float64(rng)/math.MaxUint32
	}
	var buf [4]byte
	for i := range left {
		for c, x := range [2]float32{left[i], right[i]} {
			v := math.Round(float64(x)*32767 + dither())
			binary.LittleEndian.PutUint16(buf[2*c:], uint16(int16(max(-32768, min(32767, v)))))
		}
		if _, err := w.Write(buf[:]); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
