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
}
