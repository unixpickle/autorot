package autorot

import (
	"image"
	"image/color"
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
			newX := math.Cos(angle)*xOff + math.Sin(angle)*yOff + width/2
			newY := math.Cos(angle)*yOff - math.Sin(angle)*xOff + height/2
			newImage.Set(x, y, interpolate(img, newX, newY))
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

func interpolate(img image.Image, x, y float64) color.Color {
	x1 := int(x)
	x2 := int(x + 1)
	y1 := int(y)
	y2 := int(y + 1)
	amountX1 := float64(x2) - x
	amountY1 := float64(y2) - y
	clipRange(0, img.Bounds().Dx(), &x1, &x2)
	clipRange(0, img.Bounds().Dy(), &y1, &y2)

	a11 := amountX1 * amountY1
	r11, g11, b11, _ := img.At(x1, y1).RGBA()
	a12 := amountX1 * (1 - amountY1)
	r12, g12, b12, _ := img.At(x1, y2).RGBA()
	a21 := (1 - amountX1) * amountY1
	r21, g21, b21, _ := img.At(x2, y1).RGBA()
	a22 := (1 - amountX1) * (1 - amountY1)
	r22, g22, b22, _ := img.At(x2, y2).RGBA()

	return color.RGBA{
		R: interpolateColor(r11, r12, r21, r22, a11, a12, a21, a22),
		G: interpolateColor(g11, g12, g21, g22, a11, a12, a21, a22),
		B: interpolateColor(b11, b12, b21, b22, a11, a12, a21, a22),
		A: 0xff,
	}
}

func clipRange(min, max int, vals ...*int) {
	for _, v := range vals {
		if *v < min {
			*v = min
		}
		if *v >= max {
			*v = max - 1
		}
	}
}

func interpolateColor(v1, v2, v3, v4 uint32, a1, a2, a3, a4 float64) uint8 {
	return uint8((float64(v1)*a1 + float64(v2)*a2 + float64(v3)*a3 +
		float64(v4)*a4) / 0x100)
}
