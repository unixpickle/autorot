package autorot

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec/anyvec32"
)

// A Sample stores a training sample for a network.
type Sample struct {
	// Path is the path of the unrotated image.
	Path string

	// Angle is the angle at which the image should be
	// rotated before being fed to the network.
	Angle float64
}

// A SampleList is an anyff.SampleList for rotated image
// samples.
//
// It is designed to work with data downloaded via
// https://github.com/unixpickle/imagenet.
type SampleList struct {
	Samples   []Sample
	ImageSize int
}

// ReadSampleList walks the directory and creates a sample
// for each of the images (with a random rotation).
func ReadSampleList(imageSize int, dir string) (*SampleList, error) {
	res := &SampleList{ImageSize: imageSize}
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		ext := filepath.Ext(path)
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" {
			sample := Sample{Path: path, Angle: randomAngle()}
			res.Samples = append(res.Samples, sample)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return res, nil
}

// Len returns the number of samples in the set.
func (s *SampleList) Len() int {
	return len(s.Samples)
}

// Swap swaps two sample indices.
func (s *SampleList) Swap(i, j int) {
	s.Samples[i], s.Samples[j] = s.Samples[j], s.Samples[i]
}

// GetSample generates a rotated and scaled image tensor
// for the given sample index.
func (s *SampleList) GetSample(idx int) (*anyff.Sample, error) {
	sample := s.Samples[idx]
	f, err := os.Open(sample.Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	rotated := Rotate(img, sample.Angle, s.ImageSize)
	outVec := []float32{float32(sample.Angle)}
	inVec := netInputTensor(rotated)
	return &anyff.Sample{
		Input:  anyvec32.MakeVectorData(inVec),
		Output: anyvec32.MakeVectorData(outVec),
	}, nil
}

// Slice returns a subset of the list.
func (s *SampleList) Slice(i, j int) anysgd.SampleList {
	return &SampleList{
		ImageSize: s.ImageSize,
		Samples:   append([]Sample{}, s.Samples[i:j]...),
	}
}

func randomAngle() float64 {
	return float64(rand.Intn(4)) * math.Pi / 2
}

func netInputTensor(img image.Image) []float32 {
	size := img.Bounds().Dx()
	res := make([]float32, size*size*3)

	subIdx := 0
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			pixel := img.At(x+img.Bounds().Min.X, y+img.Bounds().Min.Y)
			r, g, b, _ := pixel.RGBA()
			res[subIdx] = float32(r) / 0xffff
			res[subIdx+1] = float32(g) / 0xffff
			res[subIdx+2] = float32(b) / 0xffff
			subIdx += 3
		}
	}

	return res
}
