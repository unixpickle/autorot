package autorot

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestNetworkCost(t *testing.T) {
	t.Run("RawAngle", func(t *testing.T) {
		actual := anydiff.NewConst(
			anyvec32.MakeVectorData([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}),
		)
		desired := anydiff.NewConst(
			anyvec32.MakeVectorData([]float32{5, 2, 1, 3, 7, 4, 6, 9, 8}),
		)
		net := &Net{OutputType: RawAngle}
		actualCost := net.Cost(desired, actual, 9).Output().Data().([]float32)
		expectedCost := []float32{
			1.653643620863612,
			0.000000000000000,
			1.416146836547142,
			0.459697694131860,
			1.416146836547142,
			1.416146836547142,
			0.459697694131860,
			0.459697694131860,
			0.459697694131860,
		}
		for i, x := range expectedCost {
			a := actualCost[i]
			if math.Abs(float64(x-a)) > 1e-3 {
				t.Errorf("output %d: should be %f but got %f", i, x, a)
			}
		}
	})
	t.Run("RightAngles", func(t *testing.T) {
		actual := anydiff.NewConst(
			anyvec32.MakeVectorData([]float32{
				-0.69315, -1.38629, -2.30259, -1.89712,
				-0.69315, -1.38629, -2.30259, -1.89712,
			}),
		)
		desired := anydiff.NewConst(
			anyvec32.MakeVectorData([]float32{5, 2}),
		)
		net := &Net{OutputType: RightAngles}
		actualCost := net.Cost(desired, actual, 2).Output().Data().([]float32)
		expectedCost := []float32{1.89712, 1.38629}
		for i, x := range expectedCost {
			a := actualCost[i]
			if math.Abs(float64(x-a)) > 1e-3 {
				t.Errorf("output %d: should be %f but got %f", i, x, a)
			}
		}
	})
}

func TestRightAngleMaxes(t *testing.T) {
	vec := anyvec32.MakeVectorData([]float32{
		-0.69315, -1.38629, -2.30259, -1.89712,
		-1.20397, -0.69315, -2.30259, -2.30259,
		-1.60944, -2.30259, -2.30259, -0.51083,
	})
	actualAngles, actualProbs := rightAngleMaxes(vec)
	actual := append(actualAngles.Data().([]float32),
		actualProbs.Data().([]float32)...)
	expected := []float32{0, math.Pi / 2, 3 * math.Pi / 2, 0.5, 0.5, 0.6}
	for i, x := range expected {
		a := actual[i]
		if math.Abs(float64(x-a)) > 1e-3 {
			t.Errorf("output %d: expected %f but got %f", i, x, a)
		}
	}
}
