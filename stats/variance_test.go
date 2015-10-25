package stats_test

import (
	"testing"

	"github.com/mawicks/DragonBlood/stats"
)

func TestVarianceAccumulator(test *testing.T) {
	a := stats.NewVarianceAccumulator()
	x := []float64{1.0, 2.0, 0.0, 5.0}
	mu := []float64{1.0, 1.5, 1.0, 2.0}
	s := []float64{0.0, 0.5, 2.0, 14.0}

	for i, x := range x {
		sse, mean := a.Add(x)
		if a.Count() != i+1 {
			test.Errorf("Count() returned %d; expected %d", a.Count(), i+1)
		}

		if a.Value() != sse {
			test.Errorf("Value() returned %v but Add() returned %v", a.Value(), sse)
		}
		if sse != s[i] {
			test.Errorf("Expected sse of %v; got %v", s[i], sse)
		}
		if a.Variance() != s[i]/float64(i+1) {
			test.Errorf("Expected sse of %v; got %v", s[i]/float64(i+1), a.Variance())
		}
		if mean != mu[i] {
			test.Errorf("Expected mean of %v; got %v", s[i], mean)
		}
	}

	for i := len(x) - 1; i >= 0; i-- {
		sse, mean := a.Subtract(x[i])

		if a.Count() != i {
			test.Errorf("Count() returned %d; expected %d", a.Count(), i)
		}
		if i > 0 {
			if a.Value() != sse {
				test.Errorf("Value() returned %v but Add() returned %v", a.Value(), sse)
			}
			if sse != s[i-1] {
				test.Errorf("Expected sse of %v; got %v", s[i-1], a.Value())
			}
			if mean != mu[i-1] {
				test.Errorf("Expected mean of %v; got %v", mu[i-1], a.Value())
			}
			if a.Variance() != s[i-1]/float64(i) {
				test.Errorf("Expected sse of %v; got %v", s[i]/float64(i+1), a.Variance())
			}
		} else {
			if sse != 0.0 || a.Count() != 0 || a.Variance() != 0.0 {
				test.Errorf("Error is %v; count is %v; MSE is %v but all should be zero", sse, a.Count(), a.Variance())
			}
		}
	}
}

func TestVariance(test *testing.T) {
	x := []float64{1.0, 2.0, 0.0, 5.0}
	variance := stats.Variance(x)
	expected := 3.5
	if variance != expected {
		test.Errorf("Expected variance of %g; got %g\n", expected, variance)
	}
}
