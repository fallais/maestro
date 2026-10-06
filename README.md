# Maestro

Desktop music practice tool (Wails: Go backend + React frontend). Splits a song
into drums / bass / other / vocals with Demucs (htdemucs), running natively via
onnxruntime.

Status: separation pipeline done and verified; multitrack player, notes
(basic-pitch) and chords are next.

## Setup (Linux)

1. **System packages** (Wails needs the GTK/WebKitGTK headers; Ubuntu 24.04+ ships WebKitGTK 4.1):

   ```sh
   sudo apt install libgtk-3-dev libwebkit2gtk-4.1-dev ffmpeg
   go install github.com/wailsapp/wails/v2/cmd/wails@v2.16.0
   ```

2. **onnxruntime 1.29.1** (must match the C API version of `onnxruntime_go`):

   ```sh
   mkdir -p third_party/onnxruntime && cd third_party/onnxruntime
   curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.29.1/onnxruntime-linux-x64-1.29.1.tgz | tar xz
   ```

   Or point `MAESTRO_ORT_LIB` at an existing `libonnxruntime.so.1.29.1`.

3. **The htdemucs ONNX model** — the one manual step. There is no official ONNX
   release, so it is exported from Meta's pretrained checkpoint (MIT), once:

   ```sh
   uv venv ~/.cache/maestro/venv --python 3.12
   VIRTUAL_ENV=~/.cache/maestro/venv uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
   VIRTUAL_ENV=~/.cache/maestro/venv uv pip install demucs onnx onnxruntime onnxscript onnxslim soundfile numpy
   ~/.cache/maestro/venv/bin/python tools/export_htdemucs.py models/htdemucs.onnx
   ```

   The script checks the exported graph against the PyTorch forward pass.
   STFT/iSTFT use complex tensors that ONNX cannot express, so the graph stops
   at the spectrogram and `internal/demucs/dsp.go` does the rest.

   Keep the venv outside the repo: torch has ~20k files and exhausts the file
   watchers of `wails dev`.

   The app looks for the model in `$MAESTRO_MODEL`, `~/.local/share/maestro/models/`,
   `<exe dir>/models/`, then `./models/`.

## Run

```sh
wails dev -tags webkit2_41
wails build -tags webkit2_41   # → build/bin/maestro
```

Separated stems are cached per song (by content hash) in
`~/.local/share/maestro/tracks/`, so each song is separated once.

## Tests

```sh
~/.cache/maestro/venv/bin/python tools/make_reference.py   # fixtures from PyTorch demucs
go test ./...
```

`internal/demucs` checks the Go STFT/iSTFT and the full chunked separation
against PyTorch `demucs.apply_model` on a synthetic 12 s clip (currently
82–98 dB SNR on every stem, i.e. numerically identical).
