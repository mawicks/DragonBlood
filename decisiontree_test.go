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

	continuousTarget := db.NewNumericFeature(nil)
	continuousTarget.Add(3, 0, 3, 1, 7, 6, 5, -1)

	categoricalTarget := db.NewNumericFeature(nil)
	categoricalTarget.Add(1, 0, 1, 0, 1, 1, 1, 0)

	dtFeatures := []db.OrderedFeature{}
	for _, f := range []*db.NumericFeature{x, y, z} {
		dtFeatures = append(dtFeatures, f)
	}

	dt := db.NewDecisionTree()

	// Build a regression tree with the continuous target
	dt.Fit(dtFeatures, continuousTarget, db.NewMSECriterionFactory())
	fmt.Printf("%v\n", dt.Importances())

	tEstimate := dt.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != continuousTarget.Len() {
		test.Errorf("dt.Predict() returned result of length: %d; expected %d", len(tEstimate), continuousTarget.Len())
	}

	for i, te := range tEstimate {
		if te != continuousTarget.Value(i) {
			test.Errorf("Row %d: predicted %v; actual %v", i, te, continuousTarget.Value(i))
		}
	}

	tEstimate = dt.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != continuousTarget.Len() {
		test.Errorf("dt.Predict() returned result of length: %d; expected %d", len(tEstimate), continuousTarget.Len())
	}

	for i, te := range tEstimate {
		if te != continuousTarget.Value(i) {
			test.Errorf("Row %d: predicted %v; actual %v", i, te, continuousTarget.Value(i))
		}
	}

	// Grow a tree using categorical target with MSE criterion
	dt.Fit(dtFeatures, categoricalTarget, db.NewGiniCriterionFactory())
	fmt.Printf("%v\n", dt.Importances())

	// Repeat using categorical target with Gini criterion
	dt.Fit(dtFeatures, categoricalTarget, db.NewGiniCriterionFactory())
	fmt.Printf("%v\n", dt.Importances())

	// Repeat using categoricalTarget and entropy criterion
	dt.Fit(dtFeatures, categoricalTarget, db.NewEntropyCriterionFactory())
	fmt.Printf("%v\n", dt.Importances())

	tEstimate = dt.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != categoricalTarget.Len() {
		test.Errorf("dt.Predict() returned result of length: %d; expected %d", len(tEstimate), continuousTarget.Len())
	}

	for i, te := range tEstimate {
		if te != categoricalTarget.Value(i) {
			test.Errorf("Row %d: predicted %v; actual %v", i, te, continuousTarget.Value(i))
		}
	}
}
