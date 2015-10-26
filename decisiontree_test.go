package DragonBlood_test

import (
	"fmt"
	"testing"

	db "github.com/mawicks/DragonBlood"
)

func TestDecisionTree(test *testing.T) {
	x := db.NewNumericFeature(nil)
	x.Add(0, 1, 2, 3, 4, 5, 6, 7)

	y := db.NewNumericFeature(nil)
	y.Add(1, 2, 1, 2, 1, 2, 1, 2)

	z := db.NewNumericFeature(nil)
	z.Add(1, 1, 1, 1, 2, 2, 2, 2)

	t := db.NewNumericFeature(nil)
	t.Add(3, 0, 3, 1, 7, 6, 5, -1)

	dtFeatures := []db.OrderedFeature{}
	for _, f := range []*db.NumericFeature{x, y, z} {
		dtFeatures = append(dtFeatures, f)
	}

	dt := db.NewDecisionTree()
	af := db.NewMSEMetricFactory(1)

	dt.Fit(dtFeatures, t, af)

	tEstimate := dt.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != t.Len() {
		test.Errorf("dt.Predict() returned result of length: %d; expected %d", len(tEstimate), t.Len())
	}

	for i, te := range tEstimate {
		if te != t.Value(i) {
			test.Errorf("Row %d: predicted %v; actual %v", i, te, t.Value(i))
		}
	}

	fmt.Printf("%v\n", dt.Importances())
}
