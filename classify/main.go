package main

import (
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/unixpickle/autorot"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func main() {
	var dirPath string
	var netPath string
	flag.StringVar(&dirPath, "dir", "", "image directory")
	flag.StringVar(&netPath, "net", "", "network path")
	flag.Parse()
	if dirPath == "" || netPath == "" {
		essentials.Die("Required flags: -net and -dir. See -help for more.")
	}

	var net *autorot.Net
	if err := serializer.LoadAny(netPath, &net); err != nil {
		essentials.Die("Load network failed:", err)
	}

	outWriter := csv.NewWriter(os.Stdout)
	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		ext := strings.ToLower(filepath.Ext(info.Name()))
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" {
			if err := processImage(outWriter, net, path); err != nil {
				fmt.Fprintln(os.Stderr, err)
			}
		}
		return nil
	})

	if err != nil {
		essentials.Die("Directory listing failed:", err)
	}
}

func processImage(w *csv.Writer, network *autorot.Net, imgPath string) error {
	f, err := os.Open(imgPath)
	if err != nil {
		return errors.New("process image: " + err.Error())
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return errors.New("process image " + imgPath + ": " + err.Error())
	}
	angle, confidence := network.Evaluate(img)
	w.Write([]string{imgPath, fmt.Sprintf("%f", angle), fmt.Sprintf("%f", confidence)})
	w.Flush()
	return nil
}
