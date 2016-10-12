package autorot

import (
	"math"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// AngleCost is a cost function for angles measured
// in radians.
// It takes wrap-around into account, providing a
// smooth continuous way to reward equivalent angles.
type AngleCost struct{}

// Cost returns the sum of the discrepencies between
// actual and expected angles.
// It is minimized when the angles are all equivalent.
func (_ AngleCost) Cost(exp linalg.Vector, act autofunc.Result) autofunc.Result {
	return autofunc.Pool(act, func(act autofunc.Result) autofunc.Result {
		sin := make(linalg.Vector, len(exp))
		cos := make(linalg.Vector, len(exp))
		for i, x := range exp {
			sin[i] = math.Sin(x)
			cos[i] = math.Cos(x)
		}
		actSin := autofunc.Sin{}.Apply(act)
		actCos := autofunc.Cos{}.Apply(act)
		sinVar := &autofunc.Variable{Vector: sin}
		cosVar := &autofunc.Variable{Vector: cos}
		cost := autofunc.SumAll(autofunc.Mul(actSin, sinVar))
		cost = autofunc.Add(cost, autofunc.SumAll(autofunc.Mul(actCos, cosVar)))
		return autofunc.Scale(cost, -1)
	})
}

// CostR is like Cost but with r-operators.
func (_ AngleCost) CostR(rv autofunc.RVector, exp linalg.Vector, act autofunc.RResult) autofunc.RResult {
	return autofunc.PoolR(act, func(act autofunc.RResult) autofunc.RResult {
		sin := make(linalg.Vector, len(exp))
		cos := make(linalg.Vector, len(exp))
		for i, x := range exp {
			sin[i] = math.Sin(x)
			cos[i] = math.Cos(x)
		}
		actSin := autofunc.Sin{}.ApplyR(rv, act)
		actCos := autofunc.Cos{}.ApplyR(rv, act)
		sinVar := autofunc.NewRVariable(&autofunc.Variable{Vector: sin}, rv)
		cosVar := autofunc.NewRVariable(&autofunc.Variable{Vector: cos}, rv)
		cost := autofunc.SumAllR(autofunc.MulR(actSin, sinVar))
		cost = autofunc.AddR(cost, autofunc.SumAllR(autofunc.MulR(actCos, cosVar)))
		return autofunc.ScaleR(cost, -1)
	})
}
