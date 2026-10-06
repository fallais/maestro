package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const ortLibName = "libonnxruntime.so.1.29.1" // must match onnxruntime_go's C API version

// firstExisting returns the first path that exists, or an error listing them all.
func firstExisting(what, hint string, candidates ...string) (string, error) {
	for _, p := range candidates {
		if p == "" {
			continue
		}
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("%s not found (looked in: %s). %s", what, strings.Join(nonEmpty(candidates), ", "), hint)
}

func nonEmpty(s []string) []string {
	var out []string
	for _, v := range s {
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}

func exeDir() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	return filepath.Dir(exe)
}

// dataDir is where models and per-song caches live (~/.local/share/maestro).
func dataDir() (string, error) {
	if d := os.Getenv("XDG_DATA_HOME"); d != "" {
		return filepath.Join(d, "maestro"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".local", "share", "maestro"), nil
}

func findORTLib() (string, error) {
	return firstExisting("onnxruntime library", "Set MAESTRO_ORT_LIB or see README.",
		os.Getenv("MAESTRO_ORT_LIB"),
		filepath.Join(exeDir(), "lib", ortLibName),
		filepath.Join("third_party", "onnxruntime", "onnxruntime-linux-x64-1.29.1", "lib", ortLibName),
	)
}

func findModel(name string) (string, error) {
	data, _ := dataDir()
	return firstExisting(name+" model", "Export it with tools/export_htdemucs.py or set MAESTRO_MODEL (see README).",
		os.Getenv("MAESTRO_MODEL"),
		filepath.Join(data, "models", name+".onnx"),
		filepath.Join(exeDir(), "models", name+".onnx"),
		filepath.Join("models", name+".onnx"),
	)
}

// availableMemory reads MemAvailable from /proc/meminfo (Linux). ok is false
// when it cannot be determined.
func availableMemory() (bytes uint64, ok bool) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, false
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		if rest, found := strings.CutPrefix(s.Text(), "MemAvailable:"); found {
			kb, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimSpace(rest), " kB"), 10, 64)
			return kb * 1024, err == nil
		}
	}
	return 0, false
}

var errNoTrack = errors.New("no track loaded")
