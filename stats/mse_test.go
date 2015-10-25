package stats_test

import (
	"testing"

	"github.com/mawicks/DragonBlood/stats"
)

func TestSSEAccumulator(test *testing.T) {
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
		if a.MSE() != s[i]/float64(i+1) {
			test.Errorf("Expected sse of %v; got %v", s[i]/float64(i+1), a.MSE())
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
			if a.MSE() != s[i-1]/float64(i) {
				test.Errorf("Expected sse of %v; got %v", s[i]/float64(i+1), a.MSE())
			}
		} else {
			if sse != 0.0 || a.Count() != 0 || a.MSE() != 0.0 {
				test.Errorf("Error is %v; count is %v; MSE is %v but all should be zero", sse, a.Count(), a.MSE())
			}
		}
	}
}
