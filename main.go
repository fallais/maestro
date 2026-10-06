package main

import (
	"embed"
	"flag"
	"log"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
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
		BackgroundColour: &options.RGBA{R: 18, G: 18, B: 22, A: 1},
		OnStartup:        app.startup,
		OnShutdown:       app.shutdown,
		Bind:             []any{app},
	})
	if err != nil {
		log.Fatal(err)
	}
}
