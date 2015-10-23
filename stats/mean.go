package stats

// MeanAccumulator is a recursive accumulator for the mean.  The
// Subtract() call is not an exact inverse of the Add() unless the
// values in the series are integers.  MeanAccumulator is useful for
// evaluating candidate splits at high speed by allowing different
// subsets to be considered without re-evaluating the entire set.
// Adding a point and then subtracting it will accrue additional
// roundoff error.  If a precise mean necessary, it should be computed
// on the final selection of points using only Add() calls.
type MeanAccumulator struct {
	count int
	sum   float64
}

func NewMeanAccumulator() *MeanAccumulator {
	return &MeanAccumulator{}
}

func (a *MeanAccumulator) Add(x float64) float64 {
	// As much as I prefer the alternative formula, it doesn't
	// work well for the frequent case that x is an integer and
	// Subtract() is called.
	a.sum += x
	a.count += 1

	return a.sum
}

func (a *MeanAccumulator) Mean() float64 {
	return a.sum / float64(a.count)
}

func (a *MeanAccumulator) Value() float64 {
	return a.sum / float64(a.count)
}

func (a *MeanAccumulator) Count() int {
	return a.count
}

func (a *MeanAccumulator) Subtract(x float64) float64 {
	if a.count == 0 {
		panic("Subtract() called more than Add()")
	} else if a.count > 1 {
		a.count -= 1
		a.sum -= x
	} else {
		a.Reset()
	}

	return a.sum
}

func (a *MeanAccumulator) Reset() {
	a.sum = 0.0
	a.count = 0
}
