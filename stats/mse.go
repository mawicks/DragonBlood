package stats

// SSEAccumulator accumulates the mean and sum of squared error.  It
// is a recursive implementation that is numerically stable.  In
// particular, the squared error is guaranteed to be non-negative if
// points are added via Add() and nothing points are removed via
// Subtract().
type SSEAccumulator struct {
	mean            float64
	sumSquaredError float64
	count           int
}

func NewSSEAccumulator() *SSEAccumulator {
	return &SSEAccumulator{}
}

func (a *SSEAccumulator) Add(x float64) (sse, mean float64) {
	e := x - a.mean
	a.mean += e / float64(a.count+1)
	a.sumSquaredError += e * e * float64(a.count) / float64(a.count+1)
	a.count += 1

	return a.sumSquaredError, a.mean
}

func (a *SSEAccumulator) Mean() float64 {
	return a.mean
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

		e := x - a.mean
		a.mean -= e / float64(a.count)

		e = x - a.mean
		a.sumSquaredError -= e * e * float64(a.count) / float64(a.count+1)
	} else {
		a.Reset()
	}

	return a.sumSquaredError, a.mean
}

func (a *SSEAccumulator) Reset() {
	a.mean = 0.0
	a.sumSquaredError = 0.0
	a.count = 0
}
