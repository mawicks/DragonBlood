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
		v := a.Add(x)
		if a.Value() != v {
			test.Errorf("Value() returned %v but Add() returned %v", a.Value(), v)
		}
		if a.Value() != s[i] {
			test.Errorf("Expected sse of %v; got %v", s[i], a.Value())
		}
		if a.Mean() != mu[i] {
			test.Errorf("Expected mean of %v; got %v", s[i], a.Mean())
		}
	}

	for i := len(x) - 1; i >= 0; i-- {
		v := a.Subtract(x[i])

		if i > 0 {
			if a.Value() != v {
				test.Errorf("Value() returned %v but Add() returned %v", a.Value(), v)
			}
			if a.Value() != s[i-1] {
				test.Errorf("Expected sse of %v; got %v", s[i-1], a.Value())
			}
			if a.Mean() != mu[i-1] {
				test.Errorf("Expected mean of %v; got %v", mu[i-1], a.Value())
			}
		} else {
			if v != 0.0 || a.Value() != 0.0 || a.Mean() != 0.0 {
				fmt.Printf("Subtract() returned %v; Value() returned %v; and Mean() returned %v but all should be zero", v, a.Value(), a.Mean())
			}
		}
	}
}
