// Command run runs an autorot network on an image.
package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"

	"github.com/unixpickle/autorot"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "<net_file> <image>")
		os.Exit(1)
	}
	netFile := os.Args[1]
	imageFile := os.Args[2]

	network, err := autorot.LoadNetwork(netFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load network:", err)
		os.Exit(1)
	}

	f, err := os.Open(imageFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to open image:", err)
		os.Exit(1)
	}
	image, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to decode image:", err)
		os.Exit(1)
	}

	angle := network.Evaluate(image)
	fmt.Println(angle, "radians =", angle*180/math.Pi, "degrees")
}
