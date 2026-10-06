"""Export Meta's pretrained htdemucs to ONNX for onnxruntime-web.

torch.onnx cannot trace HTDemucs's STFT/iSTFT (complex tensors), so the exported
graph stops at the complex boundary:

    inputs   mix   [1, 2, 343980]         stereo segment, 44.1 kHz (track-normalized)
             spec  [1, 4, 2048, 336]      CAC spectrogram of `mix` (re/im per channel)
    outputs  spec_out [1, 4, 4, 2048, 336]  per-stem CAC spectrogram (de-normalized)
             wave_out [1, 4, 2, 343980]     per-stem time-branch waveform (de-normalized)

The app computes `spec` with an STFT, then each stem is iSTFT(spec_out) + wave_out.
See internal/demucs/dsp.go for the matching Go implementation.

Usage: python tools/export_htdemucs.py models/htdemucs.onnx
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
import torch
from demucs.pretrained import get_model
from onnx import numpy_helper

SEGMENT = 343980  # int(7.8 s * 44100), the htdemucs training segment
FRAMES = 336  # ceil(SEGMENT / hop 1024)
BINS = 2048  # n_fft 4096 / 2, Nyquist bin dropped


class SpectralBoundary(torch.nn.Module):
    """Runs HTDemucs.forward with its four complex-number helpers swapped out.

    `_spec`/`_magnitude` return the precomputed CAC spectrogram, `_mask` captures
    the de-normalized spectral output, `_ispec` returns a scalar zero, so forward() returns
    the time branch alone. Everything between is the unmodified upstream forward.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mix, spec):
        model = self.model
        captured = {}

        def mask(_z, m):
            captured["spec"] = m
            return m

        model._spec = lambda _x: spec
        model._magnitude = lambda z: z
        model._mask = mask
        model._ispec = lambda z, length: torch.zeros(())
        wave = model(mix)
        return captured["spec"], wave


def main(out_path):
    bag = get_model("htdemucs")
    model = bag.models[0].eval()
    assert int(model.segment * model.samplerate) == SEGMENT
    assert model.sources == ["drums", "bass", "other", "vocals"], model.sources
    assert model.cac and model.nfft == 4096 and model.hop_length == 1024

    # The fused MHA kernel (eval + no_grad) has no ONNX symbolic.
    torch.backends.mha.set_fastpath_enabled(False)
    wrapper = SpectralBoundary(model).eval()
    mix = torch.randn(1, 2, SEGMENT)
    with torch.no_grad():
        z = model.__class__._spec(model, mix)
        spec = model.__class__._magnitude(model, z)
        torch.onnx.export(
            wrapper,
            (mix, spec),
            out_path,
            input_names=["mix", "spec"],
            output_names=["spec_out", "wave_out"],
            opset_version=17,
            dynamo=False,
        )
        ref = bag.models[0].__class__.forward(_fresh(model), mix)

    # Fold shape-derived subgraphs (positional embeddings: Range/Sin/ScatterND...);
    # the WebGPU EP cannot run ScatterND, and folding makes them plain constants.
    slim = onnxslim.slim(onnx.load(out_path))
    shrink_broadcast_constants(slim)
    onnx.save(slim, out_path)
    onnx.checker.check_model(out_path, full_check=True)
    print("ops:", sorted({n.op_type for n in onnx.load(out_path).graph.node}))

    # Parity: ONNX graph + PyTorch STFT/iSTFT vs the untouched PyTorch forward.
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    spec_out, wave_out = sess.run(None, {"mix": mix.numpy(), "spec": spec.numpy()})
    zout = model.__class__._mask(_fresh(model), z, torch.from_numpy(spec_out))
    full = model.__class__._ispec(model, zout, SEGMENT) + torch.from_numpy(wave_out)
    err = (full - ref).abs().max().item()
    print(f"max abs diff vs pytorch forward: {err:.3e}  (ref peak {ref.abs().max():.3f})")
    assert err < 1e-3, err


ELEMENTWISE = {"Add", "Sub", "Mul", "Div"}


def shrink_broadcast_constants(model):
    """Undo `expand_as` materialized by constant folding (33 MB for freq_emb alone).

    An initializer used only by broadcasting ops and constant along an axis can be
    sliced to size 1 on that axis without changing any output.
    """
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, set()).add(node.op_type)
    for i, init in enumerate(model.graph.initializer):
        if not consumers.get(init.name, {"?"}) <= ELEMENTWISE:
            continue
        a = numpy_helper.to_array(init)
        if a.size < 1 << 16:
            continue
        for axis in range(a.ndim):
            if a.shape[axis] > 1 and np.all(a == a.take([0], axis=axis)):
                a = a.take([0], axis=axis)
        if a.size < np.prod(init.dims):
            model.graph.initializer[i].CopyFrom(numpy_helper.from_array(a, init.name))
    # Stale value_info would pin the old shapes.
    del model.graph.value_info[:]


def _fresh(model):
    """Drop the instance-level monkeypatches so class methods run unmodified."""
    for name in ("_spec", "_magnitude", "_mask", "_ispec"):
        model.__dict__.pop(name, None)
    return model


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/htdemucs.onnx")
