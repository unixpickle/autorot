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
//
// The outSize argument specifies the side length of the
// resulting image.
func Rotate(img image.Image, angle float64, outSize int) image.Image {
	cos := math.Cos(angle)
	sin := math.Sin(angle)

	width := float64(img.Bounds().Dx())
	height := float64(img.Bounds().Dy())
	axisBasis := &linalg.Matrix{
		Rows: 2,
		Cols: 2,
		Data: []float64{
			cos * width / 2, -sin * height / 2,
			sin * width / 2, cos * height / 2,
		},
	}

	inv := ludecomp.Decompose(axisBasis)
	var sideLength float64
	for rectFits(inv, sideLength+1) {
		sideLength++
	}

	scale := sideLength / float64(outSize)

	inImage := newRGBACache(img)
	newImage := image.NewRGBA(image.Rect(0, 0, int(outSize), int(outSize)))
	for x := 0; x < int(outSize); x++ {
		for y := 0; y < int(outSize); y++ {
			xOff := scale*float64(x) - sideLength/2
			yOff := scale*float64(y) - sideLength/2
			newX := cos*xOff + sin*yOff + width/2
			newY := cos*yOff - sin*xOff + height/2
			newImage.SetRGBA(x, y, interpolate(inImage, newX, newY))
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

func interpolate(img *rgbaCache, x, y float64) color.RGBA {
	x1 := int(x)
	x2 := int(x + 1)
	y1 := int(y)
	y2 := int(y + 1)
	amountX1 := float64(x2) - x
	amountY1 := float64(y2) - y
	clipRange(0, img.Width(), &x1, &x2)
	clipRange(0, img.Height(), &y1, &y2)

	a11 := amountX1 * amountY1
	r11, g11, b11 := img.At(x1, y1)
	a12 := amountX1 * (1 - amountY1)
	r12, g12, b12 := img.At(x1, y2)
	a21 := (1 - amountX1) * amountY1
	r21, g21, b21 := img.At(x2, y1)
	a22 := (1 - amountX1) * (1 - amountY1)
	r22, g22, b22 := img.At(x2, y2)

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

func interpolateColor(v1, v2, v3, v4 float64, a1, a2, a3, a4 float64) uint8 {
	return uint8(v1*a1 + v2*a2 + v3*a3 + v4*a4)
}

type rgbaCache struct {
	img        image.Image
	cache      [][3]float64
	cacheValid []bool
	bounds     image.Rectangle
}

func newRGBACache(img image.Image) *rgbaCache {
	pixels := img.Bounds().Dx() * img.Bounds().Dy()
	return &rgbaCache{
		img:        img,
		cache:      make([][3]float64, pixels),
		cacheValid: make([]bool, pixels),
		bounds:     img.Bounds(),
	}
}

func (r *rgbaCache) Width() int {
	return r.bounds.Dx()
}

func (r *rgbaCache) Height() int {
	return r.bounds.Dy()
}

func (r *rgbaCache) At(x, y int) (float64, float64, float64) {
	idx := x + y*r.bounds.Dx()
	if r.cacheValid[idx] {
		c := r.cache[idx]
		return c[0], c[1], c[2]
	}
	r.cacheValid[idx] = true
	rInt, gInt, bInt, _ := r.img.At(x+r.bounds.Min.X, y+r.bounds.Min.Y).RGBA()
	r.cache[idx] = [3]float64{
		float64(rInt) / 0x100,
		float64(gInt) / 0x100,
		float64(bInt) / 0x100,
	}
	c := r.cache[idx]
	return c[0], c[1], c[2]
}
