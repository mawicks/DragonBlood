package stats

// VarianceAccumulator is an accumulator for the mean and variance error
// (primarily the variance, where the mean is accumluated as a
// by-product).  It is a recursive implementation that is numerically
// stable if Subtract() is not called.  In particular, the squared
// error is guaranteed to be non-negative if points are added via
// Add() and no points are removed via Subtract().  Subtract() is not
// an exact inverse of the Add() call for the square error estimate.
// Subtract() is an exact inverse for the mean estimate if the values
// of the series are integers.  VarianceAccumulator is useful for evaluate
// candidate splits at high speed by allows different subsets to be
// considered without re-evaluating the entire set.  Adding a point
// and then subtracting it will accrue additional roundoff error.  If
// a precise error estimate is necessary, it should be computed on the
// final selection of points using only Add() calls.
type VarianceAccumulator struct {
	sum             float64
	sumSquaredError float64
	count           int
}

func NewVarianceAccumulator() *VarianceAccumulator {
	return &VarianceAccumulator{}
}

func (a *VarianceAccumulator) Add(x float64) (sse, mean float64) {
	if a.count > 0 {
		e := x - a.sum/float64(a.count)
		a.sumSquaredError += e * e * float64(a.count) / float64(a.count+1)
	}
	a.sum += x
	a.count += 1

	return a.sumSquaredError, a.sum / float64(a.count)
}

func (a *VarianceAccumulator) Mean() float64 {
	return a.sum / float64(a.count)
}
func (a *VarianceAccumulator) Variance() float64 {
	if a.count > 0 {
		return a.sumSquaredError / float64(a.count)
	} else {
		return 0.0
	}
}

func (a *VarianceAccumulator) Value() float64 {
	return a.sumSquaredError
}

func (a *VarianceAccumulator) Count() int {
	return a.count
}

func (a *VarianceAccumulator) Subtract(x float64) (sse, mean float64) {
	if a.count == 0 {
		panic("Subtract() called more than Add()")
	} else if a.count > 1 {
		a.count -= 1
		a.sum -= x

		e := x - a.sum/float64(a.count)
		a.sumSquaredError -= e * e * float64(a.count) / float64(a.count+1)
	} else {
		a.Reset()
	}

	return a.sumSquaredError, a.sum / float64(a.count)
}

func (a *VarianceAccumulator) Reset() {
	a.sum = 0.0
	a.sumSquaredError = 0.0
	a.count = 0
}

func Variance(sequence []float64) float64 {
	accumulator := NewVarianceAccumulator()
	for _, x := range sequence {
		accumulator.Add(x)
	}

	return accumulator.Variance()
}
