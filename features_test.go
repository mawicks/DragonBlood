package DragonBlood_test

import (
	"math"
	"testing"

	db "github.com/mawicks/DragonBlood"
)

func TestNumericFeature(t *testing.T) {
	// Assign to Feature to ensure NumericFeature implements Feature
	var f db.Feature = db.NewNumericFeature()
	// Assign it back so we can use its NumericFeature methods
	var nf = f.(*db.NumericFeature)

	for _, s := range []string{"3.0", "2.0", "non-numeric"} {
		nf.AddFromString(s)
	}

	for _, v := range []float64{1.0, 4.0} {
		nf.Add(v)
	}

	// Make sure integers work
	nf.Add(5)

	check := func(index int, expected float64) {
		actual := nf.NumericValue(index)
		altActual := nf.Value(index).(float64)
		if actual != altActual && math.IsNaN(actual) != math.IsNaN(altActual) {
			t.Errorf("NumericValue(%d) is %g; Value(%d) is %v", index, actual, index, altActual)
		}
		if actual != expected && math.IsNaN(actual) != math.IsNaN(expected) {
			t.Errorf("Get(%d) got %g; expecting %g", index, actual, expected)
		}
	}

	check(0, 3.0)
	check(1, 2.0)
	check(2, math.NaN())
	check(3, 1.0)
	check(4, 4.0)
	check(5, 5.0)

	ordercheck := func(index int, expected int) {
		orderedIndex := nf.InOrder(index)
		if orderedIndex != expected {
			t.Errorf("Get(%d) got %d; expecting %d", index, orderedIndex, expected)
		}
	}

	nf.Sort()

	ordercheck(0, 3)
	ordercheck(1, 1)
	ordercheck(2, 0)
	ordercheck(3, 4)
	ordercheck(4, 5)
	ordercheck(5, 2)

	if nf.Len() != 6 {
		t.Errorf("Len() returned %d; expecting %d", nf.Len(), 5)
	}
}

func TestCategoricalFeature(t *testing.T) {
	// Assign to Feature to ensure CategoricalFeature implements Feature
	cf := db.NewCategoricalFeature()

	testStrings := []string{"delta", "alpha", "beta", "beta", "gamma"}

	for _, s := range testStrings {
		cf.AddFromString(s)
	}

	check := func(index int, expected string) {
		actual := cf.Value(index).(string)
		altActual := cf.Decode(int(cf.NumericValue(index))).(string)
		if actual != altActual {
			t.Errorf("Value(%d) returned %v; Decode(NumericValue(%d) returned %v", index, actual, index, altActual)
		}
		if actual != expected {
			t.Errorf("Get(%d) got %v; expecting %v", index, actual, expected)
		}
	}

	if cf.Len() != len(testStrings) {
		t.Errorf("Length() returned %d; expecting %d", cf.Len(), 5)
	}

	cf.OrderEncoding()

	for i, s := range testStrings {
		check(i, s)
	}

	ordercheck := func(index int, expected int) {
		if got := int(cf.NumericValue(index)); got != expected {
			t.Errorf("Get(%d) got %d; expecting %d", index, got, expected)
		}
	}

	// "alpha", "beta", "delta", "gamma" should be encoded as
	// 0, 1, 2, 3 respectively

	ordercheck(0, 2) // "delta" should encode to 2
	ordercheck(1, 0) // "alpha" should encode to 0
	ordercheck(2, 1) // "beta" should encode to 1
	ordercheck(3, 1) // "beta" should encode to 1
	ordercheck(4, 3) // "gamma" should encode to 3

	if n := cf.Categories(); n != 4 {
		t.Errorf("Expected %d categories; got %d\n", 4, n)
	}
}
