package autorot

import (
	"image"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"

	_ "github.com/unixpickle/anynet/anyconv"
)

// OutputType specifies the output format and loss
// function for a network.
type OutputType int

const (
	RawAngle OutputType = iota
)

func init() {
	var n Net
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNet)
}

// A Net is a neural net that predicts angles from images.
type Net struct {
	// Side length of input images.
	InputSize int

	OutputType OutputType
	Net        anynet.Net
}

// DeserializeNet deserializes a Net.
func DeserializeNet(d []byte) (*Net, error) {
	var res Net
	err := serializer.DeserializeAny(d, &res.InputSize, &res.OutputType, &res.Net)
	if err != nil {
		return nil, err
	}
	return &res, nil
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
	switch n.OutputType {
	case RawAngle:
		return float64(anyvec.Sum(out).(float32))
	default:
		panic("invalid OutputType")
	}
}

// Cost computes the total cost, given the desired output
// angles and the outputs from the network.
func (n *Net) Cost(desired, actual anydiff.Res, num int) anydiff.Res {
	if num != desired.Output().Len() {
		panic("bad batch size")
	}
	switch n.OutputType {
	case RawAngle:
		return anydiff.Complement(anydiff.Cos(anydiff.Sub(actual, desired)))
	default:
		panic("invalid OutputType")
	}
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
		serializer.Int(n.OutputType),
		n.Net,
	)
}
