// Command repurpose converts an ImageNet classifier to an
// autorot network.
package main

import (
	"flag"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/autorot"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/serializer"
)

func main() {
	var inFile string
	var outFile string
	var removeLayers int

	flag.StringVar(&inFile, "in", "", "imagenet classifier path")
	flag.StringVar(&outFile, "out", "", "output network path")
	flag.IntVar(&removeLayers, "remove", 2, "number of layers to remove")

	flag.Parse()

	if inFile == "" || outFile == "" {
		essentials.Die("Required flags: -in and -out. See -help for more.")
	}

	var inNet *imagenet.Classifier
	if err := serializer.LoadAny(inFile, &inNet); err != nil {
		essentials.Die("Load input failed:", err)
	}
	if inNet.InWidth != inNet.InHeight {
		essentials.Die("Input dimensions do not form a square.")
	}

	newNet := inNet.Net[:len(inNet.Net)-removeLayers]
	zeroIn := anydiff.NewConst(anyvec32.MakeVector(inNet.InWidth * inNet.InHeight * 3))
	outCount := newNet.Apply(zeroIn, 1).Output().Len()
	newNet = append(newNet, anynet.NewFC(anyvec32.CurrentCreator(), outCount, 1))
	out := &autorot.Net{
		InputSize:  inNet.InWidth,
		OutputType: autorot.RawAngle,
		Net:        newNet,
	}
	if err := serializer.SaveAny(outFile, out); err != nil {
		essentials.Die("Save failed:", err)
	}
}
