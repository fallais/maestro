package main

// SeparationModel is a Demucs export from tools/export_htdemucs.py.
type SeparationModel struct {
	ID      string   // file name in models/, without .onnx
	Sources []string // output order, as recorded in the ONNX metadata
}

// stemModel is the model the app separates with: all six htdemucs_6s stems.
var stemModel = SeparationModel{
	ID:      "htdemucs_6s",
	Sources: []string{"drums", "bass", "other", "vocals", "guitar", "piano"},
}
