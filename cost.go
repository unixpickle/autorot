package autorot

import "github.com/unixpickle/anydiff"

// Cost computes a measure of the difference between two
// angles, taking into account the periodicity of angles.
type Cost struct{}

// Cost computes the cost.
func (c Cost) Cost(a, b anydiff.Res, n int) anydiff.Res {
	dotProducts := anydiff.Pool(a, func(a anydiff.Res) anydiff.Res {
		return anydiff.Pool(b, func(b anydiff.Res) anydiff.Res {
			return anydiff.Add(
				anydiff.Mul(anydiff.Sin(a), anydiff.Sin(b)),
				anydiff.Mul(anydiff.Cos(a), anydiff.Cos(b)),
			)
		})
	})

	// Utilize the fact that the dot product between two unit
	// vectors is < 1 when the vectors are not equal.
	differences := anydiff.Complement(dotProducts)
	mat := &anydiff.Matrix{
		Data: differences,
		Rows: n,
		Cols: differences.Output().Len() / n,
	}
	return anydiff.SumCols(mat)
}
