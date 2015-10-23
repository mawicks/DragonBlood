package stats

// SSEAccumulator is an accumulator for the mean and sum of squared
// error (primarily the sum of squared error, where the mean is
// accumluated as a by-product).  It is a recursive implementation
// that is numerically stable if Subtract() is not called.  In
// particular, the squared error is guaranteed to be non-negative if
// points are added via Add() and no points are removed via
// Subtract().  Subtract() is not an exact inverse of the Add() call
// for the square error estimate.  Subtract() is an exact inverse for
// the mean estimate if the values of the series are integers.
// SSEAccumulator is useful for evaluate candidate splits at high
// speed by allows different subsets to be considered without
// re-evaluating the entire set.  Adding a point and then subtracting
// it will accrue additional roundoff error.  If a precise error
// estimate is necessary, it should be computed on the final selection
// of points using only Add() calls.
type SSEAccumulator struct {
	sum             float64
	sumSquaredError float64
	count           int
}

func NewSSEAccumulator() *SSEAccumulator {
	return &SSEAccumulator{}
}

func (a *SSEAccumulator) Add(x float64) (sse, mean float64) {
	if a.count > 0 {
		e := x - a.sum/float64(a.count)
		a.sumSquaredError += e * e * float64(a.count) / float64(a.count+1)
	}
	a.sum += x
	a.count += 1

	return a.sumSquaredError, a.sum / float64(a.count)
}

func (a *SSEAccumulator) Mean() float64 {
	return a.sum / float64(a.count)
}
func (a *SSEAccumulator) MSE() float64 {
	if a.count > 0 {
		return a.sumSquaredError / float64(a.count)
	} else {
		return 0.0
	}
}

func (a *SSEAccumulator) Value() float64 {
	return a.sumSquaredError
}

func (a *SSEAccumulator) Count() int {
	return a.count
}

func (a *SSEAccumulator) Subtract(x float64) (sse, mean float64) {
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

func (a *SSEAccumulator) Reset() {
	a.sum = 0.0
	a.sumSquaredError = 0.0
	a.count = 0
}
