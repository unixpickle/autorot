package autorot

import (
	"image"
	"math"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/num-analysis/linalg/ludecomp"
)

// Rotate rotates an image around its center and returns
// the largest centered square cropping that does not go
// out of the rotated image's bounds.
//
// The angle is specified in clockwise radians.
func Rotate(img image.Image, angle float64) image.Image {
	width := float64(img.Bounds().Dx())
	height := float64(img.Bounds().Dy())
	axisBasis := &linalg.Matrix{
		Rows: 2,
		Cols: 2,
		Data: []float64{
			math.Cos(angle) * width / 2, -math.Sin(angle) * height / 2,
			math.Sin(angle) * width / 2, math.Cos(angle) * height / 2,
		},
	}

	inv := ludecomp.Decompose(axisBasis)
	var sideLength float64
	for rectFits(inv, sideLength+1) {
		sideLength++
	}

	// TODO: figure out how to use draw2d for this.
	newImage := image.NewRGBA(image.Rect(0, 0, int(sideLength), int(sideLength)))
	for x := 0; x < int(sideLength); x++ {
		for y := 0; y < int(sideLength); y++ {
			xOff := float64(x) - sideLength/2
			yOff := float64(y) - sideLength/2
			newX := math.Cos(angle)*xOff + math.Sin(angle)*yOff
			newY := math.Cos(angle)*yOff - math.Sin(angle)*xOff
			rawNewX := math.Min(math.Max(newX+float64(width/2), 0), width-1)
			rawNewY := math.Min(math.Max(newY+float64(height/2), 0), height-1)
			pixel := img.At(int(rawNewX+0.5), int(rawNewY+0.5))
			newImage.Set(x, y, pixel)
		}
	}

	return newImage
}

func rectFits(axisBasis *ludecomp.LU, sideLength float64) bool {
	for xScale := -1; xScale <= 1; xScale += 2 {
		for yScale := -1; yScale <= 1; yScale += 2 {
			corner := []float64{
				sideLength * float64(xScale) / 2,
				sideLength * float64(yScale) / 2,
			}
			solution := axisBasis.Solve(corner)
			if solution.MaxAbs() > 1 {
				return false
			}
		}
	}
	return true
}
