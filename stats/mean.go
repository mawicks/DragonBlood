package stats

// SSEAccumulator accumulates the mean and sum of squared error.  It
// is a recursive implementation that is numerically stable.  In
// particular, the squared error is guaranteed to be non-negative if
// points are added via Add() and nothing points are removed via
// Subtract().
type MeanAccumulator struct {
	count int
	mean  float64
}

func NewMeanAccumulator() *MeanAccumulator {
	return &MeanAccumulator{}
}

func (a *MeanAccumulator) Add(x float64) float64 {
	e := x - a.mean
	a.mean += e / float64(a.count+1)
	a.count += 1

	return a.mean
}

func (a *MeanAccumulator) Mean() float64 {
	return a.mean
}

func (a *MeanAccumulator) Value() float64 {
	return a.mean
}

func (a *MeanAccumulator) Count() int {
	return a.count
}

func (a *MeanAccumulator) Subtract(x float64) float64 {
	if a.count == 0 {
		panic("Subtract() called more than Add()")
	} else if a.count > 1 {
		a.count -= 1

		e := x - a.mean
		a.mean -= e / float64(a.count)
	} else {
		a.Reset()
	}

	return a.mean
}

func (a *MeanAccumulator) Reset() {
	a.mean = 0.0
	a.count = 0
}
