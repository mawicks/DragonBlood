package DragonBlood_test

import (
	"math"
	"testing"

	"github.com/mawicks/DragonBlood"
)

func TestNumericFeature(t *testing.T) {
	nf := DragonBlood.NewNumericFeature("foo")

	for _, s := range []string{"1.0", "2.0", "non-numeric"} {
		nf.AddFromString(s)
	}

	for _, v := range []float64{3.0, 4.0} {
		nf.Add(v)
	}

	if nf.Name() != "foo" {
		t.Errorf(`Name() failed to return "foo"`)
	}

	check := func(index int, expected float64) {
		actual := nf.Get(index).(float64)
		if actual != expected && math.IsNaN(actual) != math.IsNaN(expected) {
			t.Errorf("Get(%d) got %g; expecting %g", index, actual, expected)
		}
	}

	check(0, 1.0)
	check(1, 2.0)
	check(2, math.NaN())
	check(3, 3.0)
	check(4, 4.0)

	if nf.Length() != 5 {
		t.Errorf("Length() returned %d; expecting %d", nf.Length(), 5)
	}
}
