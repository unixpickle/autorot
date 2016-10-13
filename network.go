package autorot

import (
	"errors"
	"image"
	"io/ioutil"
	"math"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var n Network
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNetwork)
}

// A Network attempts to predict the angle at which an
// image is rotated.
type Network struct {
	// InputSize is the side length of input images.
	InputSize int

	// Net is the underlying network for classification.
	Net neuralnet.Network
}

// NewNetwork creates a new, randomly initialized and
// untrained network.
func NewNetwork(size int) *Network {
	tensorSize := size
	tensorDepth := 3
	res := &Network{
		InputSize: size,
		Net: neuralnet.Network{
			&neuralnet.RescaleLayer{Bias: -0.5, Scale: 1},
		},
	}
	for _, depth := range []int{10, 30, 60, 60} {
		conv := &neuralnet.ConvLayer{
			InputWidth:   tensorSize,
			InputHeight:  tensorSize,
			InputDepth:   tensorDepth,
			Stride:       1,
			FilterCount:  depth,
			FilterWidth:  3,
			FilterHeight: 3,
		}
		pooling := &neuralnet.MaxPoolingLayer{
			InputWidth:  conv.OutputWidth(),
			InputHeight: conv.OutputHeight(),
			InputDepth:  conv.OutputDepth(),
			XSpan:       2,
			YSpan:       2,
		}
		tensorSize = pooling.OutputWidth()
		tensorDepth = depth
		res.Net = append(res.Net, conv, neuralnet.ReLU{}, pooling)
	}
	res.Net = append(res.Net, &neuralnet.DenseLayer{
		InputCount:  tensorSize * tensorSize * tensorDepth,
		OutputCount: 50,
	}, neuralnet.HyperbolicTangent{}, &neuralnet.DenseLayer{
		InputCount:  50,
		OutputCount: 1,
	})
	res.Net.Randomize()
	for _, layer := range res.Net {
		if cn, ok := layer.(*neuralnet.ConvLayer); ok {
			for i := range cn.Biases.Vector {
				cn.Biases.Vector[i] = 1
			}
		}
	}
	return res
}

// DeserializeNetwork deserializes a network.
func DeserializeNetwork(d []byte) (*Network, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	invalidErr := errors.New("invalid Network slice")
	if len(slice) != 2 {
		return nil, invalidErr
	}
	netObj, ok1 := slice[0].(neuralnet.Network)
	sizeObj, ok2 := slice[1].(serializer.Int)
	if !ok1 || !ok2 {
		return nil, invalidErr
	}
	return &Network{
		Net:       netObj,
		InputSize: int(sizeObj),
	}, nil
}

// LoadNetwork loads a network from a file.
func LoadNetwork(path string) (*Network, error) {
	contents, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DeserializeNetwork(contents)
}

// Evaluate runs the network on an image and reports the
// angle at which it should be rotated.
func (n *Network) Evaluate(img image.Image) float64 {
	if img.Bounds().Dx() != img.Bounds().Dy() {
		// Hack to crop the center square.
		img = Rotate(img, 0)
	}

	inTensor := netInputTensor(img, n.InputSize)
	inVar := &autofunc.Variable{Vector: inTensor.Data}
	out := n.Net.Apply(inVar).Output()[0]
	for out < math.Pi {
		out += math.Pi * 2
	}
	for out > math.Pi {
		out -= math.Pi * 2
	}
	return out
}

// Save saves the network to a file.
func (n *Network) Save(path string) error {
	data, err := n.Serialize()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, data, 0755)
}

// SerializerType is the unique ID used to serialize
// Networks with the serializer package.
func (n *Network) SerializerType() string {
	return "github.com/unixpickle/autorot.Network"
}

// Serialize serializes the network.
func (n *Network) Serialize() ([]byte, error) {
	return serializer.SerializeSlice([]serializer.Serializer{
		n.Net,
		serializer.Int(n.InputSize),
	})
}
