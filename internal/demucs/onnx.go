package demucs

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	runtimeOnce sync.Once
	runtimeErr  error
)

// InitRuntime loads the onnxruntime shared library (libonnxruntime.so.1.29.x,
// onnxruntime.dll, libonnxruntime.dylib). Safe to call more than once; only the
// first path is used.
func InitRuntime(libPath string) error {
	runtimeOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		runtimeErr = ort.InitializeEnvironment()
		if runtimeErr != nil {
			runtimeErr = fmt.Errorf("loading onnxruntime from %s: %w", libPath, runtimeErr)
		}
	})
	return runtimeErr
}

// ONNXModel is a Model backed by an onnxruntime session with preallocated I/O.
// Not safe for concurrent use.
type ONNXModel struct {
	session                     *ort.AdvancedSession
	runOpts                     *ort.RunOptions
	mix, spec, specOut, waveOut *ort.Tensor[float32]
}

// LoadONNX creates a CPU session for the htdemucs graph. threads <= 0 uses all cores.
func LoadONNX(modelPath string, threads int) (*ONNXModel, error) {
	m := &ONNXModel{}
	var err error
	if m.mix, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 2, Segment)); err != nil {
		return nil, err
	}
	if m.spec, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 4, Bins, Frames)); err != nil {
		return nil, m.destroyWith(err)
	}
	if m.specOut, err = ort.NewEmptyTensor[float32](ort.NewShape(1, int64(NumSources), 4, Bins, Frames)); err != nil {
		return nil, m.destroyWith(err)
	}
	if m.waveOut, err = ort.NewEmptyTensor[float32](ort.NewShape(1, int64(NumSources), 2, Segment)); err != nil {
		return nil, m.destroyWith(err)
	}

	if m.runOpts, err = ort.NewRunOptions(); err != nil {
		return nil, m.destroyWith(err)
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, m.destroyWith(err)
	}
	defer opts.Destroy()
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	if err := opts.SetIntraOpNumThreads(threads); err != nil {
		return nil, m.destroyWith(err)
	}
	// The arena keeps peak buffers forever: 5.0 GB RSS with it, 3.1 GB without, same speed.
	if err := opts.SetCpuMemArena(false); err != nil {
		return nil, m.destroyWith(err)
	}
	if err := opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, m.destroyWith(err)
	}

	m.session, err = ort.NewAdvancedSession(modelPath,
		[]string{"mix", "spec"}, []string{"spec_out", "wave_out"},
		[]ort.Value{m.mix, m.spec}, []ort.Value{m.specOut, m.waveOut}, opts)
	if err != nil {
		return nil, m.destroyWith(fmt.Errorf("creating session for %s: %w", modelPath, err))
	}
	return m, nil
}

// Run implements Model. Cancelling ctx terminates the in-flight inference.
func (m *ONNXModel) Run(ctx context.Context, mix, spec []float32) ([]float32, []float32, error) {
	copy(m.mix.GetData(), mix)
	copy(m.spec.GetData(), spec)
	if err := m.runOpts.UnsetTerminate(); err != nil {
		return nil, nil, err
	}
	stop := context.AfterFunc(ctx, func() { m.runOpts.Terminate() })
	err := m.session.RunWithOptions(m.runOpts)
	stop()
	if ctx.Err() != nil {
		return nil, nil, ctx.Err()
	}
	if err != nil {
		return nil, nil, fmt.Errorf("htdemucs inference: %w", err)
	}
	return m.specOut.GetData(), m.waveOut.GetData(), nil
}

// Destroy releases the session and tensors.
func (m *ONNXModel) Destroy() error { return m.destroyWith(nil) }

func (m *ONNXModel) destroyWith(err error) error {
	if m.session != nil {
		m.session.Destroy()
	}
	if m.runOpts != nil {
		m.runOpts.Destroy()
	}
	for _, t := range []*ort.Tensor[float32]{m.mix, m.spec, m.specOut, m.waveOut} {
		if t != nil {
			t.Destroy()
		}
	}
	return err
}
