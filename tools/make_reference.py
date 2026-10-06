"""Synthesize a short test clip and separate it with the official PyTorch demucs.

Writes to OUT_DIR (default tools/fixtures/):
  clip.wav                  12 s synthetic song, 44.1 kHz stereo
  clip.f32                  same, raw float32 [2, N]
  ref_stems.f32             apply_model(shifts=0, overlap=0.25) output [4, 2, N]
  seg_mix.f32 / seg_spec.f32 / seg_spec_out.f32 / seg_wave_out.f32 / seg_full.f32
                            one segment's model I/O, to test the JS STFT/iSTFT alone

The song: 120 bpm, Am-F-C-G pad (other), root-note bass, kick/snare/hats, and a
vibrato "voice" melody, so separation, chords and notes all have a known answer.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

SR = 44100
SEGMENT = 343980


def synth(seconds=12.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    beat = 0.5
    drums = np.zeros(n)
    bass = np.zeros(n)
    pad = np.zeros(n)
    voice = np.zeros(n)

    def env(length, decay):
        return np.exp(-np.arange(length) / SR / decay)

    for b in range(int(seconds / beat)):
        i = int(b * beat * SR)
        k = min(int(0.3 * SR), n - i)
        tt = np.arange(k) / SR
        drums[i:i + k] += np.sin(2 * np.pi * (50 * tt + 100 * 0.04 * (1 - np.exp(-tt / 0.04)))) * env(k, 0.12)
        if b % 2 == 1:
            s = min(int(0.2 * SR), n - i)
            drums[i:i + s] += 0.5 * rng.standard_normal(s) * env(s, 0.05)
        for h in (0, 0.25):
            j = int((b * beat + h) * SR)
            s = min(int(0.05 * SR), n - j)
            if s > 0:
                drums[j:j + s] += 0.15 * np.diff(rng.standard_normal(s + 1)) * env(s, 0.015)

    # Am F C G, one chord per 2 beats... per bar (4 beats = 2 s).
    chords = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]
    roots = [33, 29, 36, 31]  # A1 F1 C2 G1
    melody = [69, 72, 71, 69, 65, 67, 69, 72, 67, 64, 67, 71, 74, 71, 67, 62]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    for bar in range(int(seconds / 2)):
        c = bar % 4
        i0, i1 = int(bar * 2 * SR), min(int((bar + 1) * 2 * SR), n)
        tt = t[i0:i1] - t[i0]
        for m in chords[c]:
            for h in range(1, 6):
                pad[i0:i1] += np.sin(2 * np.pi * hz(m) * h * tt) / h ** 1.5 * 0.08
        for q in range(4):  # bass: root on each beat
            j0 = i0 + int(q * beat * SR)
            j1 = min(j0 + int(0.45 * SR), i1)
            u = t[j0:j1] - t[j0]
            f = hz(roots[c] + (12 if q == 3 else 0))
            bass[j0:j1] += sum(np.sin(2 * np.pi * f * h * u) / h for h in range(1, 4)) * np.minimum(1, u / 0.01) * env(j1 - j0, 0.4) * 0.5
        for q in range(4):  # voice: one note per beat, vibrato, harmonics
            j0 = i0 + int(q * beat * SR)
            j1 = min(j0 + int(0.48 * SR), i1)
            u = t[j0:j1] - t[j0]
            f = hz(melody[(bar * 4 + q) % len(melody)])
            ph = 2 * np.pi * (f * u + 0.015 * f / 5.5 * np.sin(2 * np.pi * 5.5 * u))
            voice[j0:j1] += sum(np.sin(h * ph) * [1, 0.5, 0.6, 0.2, 0.1][h - 1] for h in range(1, 6)) * np.sin(np.pi * u / u[-1]) * 0.12

    left = drums + bass + 0.8 * pad + voice
    right = drums + bass + 1.2 * pad + 0.9 * voice
    mix = np.stack([left, right]).astype(np.float32)
    return mix / np.abs(mix).max() * 0.8


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mix = synth()
    sf.write(out / "clip.wav", mix.T, SR, subtype="FLOAT")
    mix.tofile(out / "clip.f32")
    print("clip", mix.shape)

    model = get_model("htdemucs").models[0].eval()
    wav = torch.from_numpy(mix)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        stems = apply_model(model, ((wav - mean) / std)[None], shifts=0, split=True, overlap=0.25)
    stems = (stems * std + mean)[0].numpy().astype(np.float32)
    stems.tofile(out / "ref_stems.f32")
    print("ref_stems", stems.shape)

    # One segment's intermediates (the first chunk, normalized like apply_model does).
    seg = ((wav - mean) / std)[:, :SEGMENT][None]
    with torch.no_grad():
        z = model._spec(seg)
        spec = model._magnitude(z)
        full = model(seg)
    sess = ort.InferenceSession(str(Path(__file__).parent.parent / "models/htdemucs.onnx"))
    spec_out, wave_out = sess.run(None, {"mix": seg.numpy(), "spec": spec.numpy()})
    with torch.no_grad():
        onnx_full = model._ispec(model._mask(z, torch.from_numpy(spec_out)), SEGMENT) + torch.from_numpy(wave_out)
    print("segment onnx vs torch max diff:", (onnx_full - full).abs().max().item())
    for name, arr in [("seg_mix", seg), ("seg_spec", spec), ("seg_spec_out", spec_out),
                      ("seg_wave_out", wave_out), ("seg_full", onnx_full)]:
        np.asarray(arr, dtype=np.float32).tofile(out / f"{name}.f32")

    for i, name in enumerate(model.sources):
        rms = np.sqrt((stems[i] ** 2).mean())
        print(f"  {name:7s} rms {rms:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tools/fixtures")
