package stats_test

import (
	"fmt"
	"testing"

	"github.com/mawicks/DragonBlood/stats"
)

func TestSSEAccumulator(test *testing.T) {
	a := stats.NewSSEAccumulator()
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
		} else {
			if sse != 0.0 || mean != 0.0 || a.Count() != 0.0 {
				fmt.Printf("Error is %v; mean is %v; count is %v but all should be zero", sse, mean, a.Count())
			}
		}
	}
}