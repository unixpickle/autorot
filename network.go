package autorot

import (
	"image"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

func init() {
	var n Net
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNet)
}

// A Net is a neural net that predicts angles from images.
type Net struct {
	// InputSize is the side length of input images.
	InputSize int

	Net anynet.Net
}

// DeserializeNet deserializes a Net.
func DeserializeNet(d []byte) (*Net, error) {
	var inSize serializer.Int
	var net anynet.Net
	if err := serializer.DeserializeAny(d, &inSize, &net); err != nil {
		return nil, err
	}
	return &Net{
		InputSize: int(inSize),
		Net:       net,
	}, nil
}

// Evaluate generates a prediction for an image.
func (n *Net) Evaluate(img image.Image) float64 {
	if img.Bounds().Dx() != img.Bounds().Dy() {
		// Hack to crop the center square.
		img = Rotate(img, 0)
	}
	inTensor := netInputTensor(img, n.InputSize)
	inConst := anydiff.NewConst(anyvec32.MakeVectorData(inTensor))
	out := n.Net.Apply(inConst, 1).Output()
	return float64(anyvec.Sum(out).(float32))
}

// SerializerType returns the unique ID used to serialize
// a Net with the serializer package.
func (n *Net) SerializerType() string {
	return "github.com/unixpickle/autorot.Net"
}

// Serialize serializes the Net.
func (n *Net) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		serializer.Int(n.InputSize),
		n.Net,
	)
}
