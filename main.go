package main

import (
	"embed"
	"flag"
	"log"
	"os"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	// Wails formats the GTK background CSS with printf("%f"); under a locale with
	// a decimal comma (fr_FR, de_DE, ...) GTK rejects "rgba(.., 1,0)" with a
	// "Theme parsing error". GTK reads the locale at init, so pin numbers to C.
	if os.Getenv("LC_ALL") == "" {
		os.Setenv("LC_NUMERIC", "C")
	}
	selfTest := flag.Bool("selftest", false, "play the given file for 1 s, log audio diagnostics and quit")
	flag.Usage = func() { log.Printf("usage: maestro [--selftest] [song.mp3]") }
	flag.Parse()
	app := NewApp(flag.Arg(0), *selfTest)
	err := wails.Run(&options.App{
		Title:     "Maestro",
		Width:     1280,
		Height:    820,
		MinWidth:  900,
		MinHeight: 600,
		AssetServer: &assetserver.Options{
			Assets:  assets,
			Handler: mediaHandler(), // /media/... stem and mix WAVs
		},
		BackgroundColour: &options.RGBA{R: 18, G: 18, B: 22, A: 255},
		OnStartup:        app.startup,
		OnShutdown:       app.shutdown,
		Bind:             []any{app},
	})
	if err != nil {
		log.Fatal(err)
	}
}
