package DragonBlood_test

import (
	"fmt"
	"log"
	"testing"

	db "github.com/mawicks/DragonBlood"
)

func TestDecisionTree(test *testing.T) {
	x := db.NewNumericFeature(0, 1, 2, 3, 4, 5, 6, 7)
	y := db.NewNumericFeature(1, 2, 1, 2, 1, 2, 1, 2)
	z := db.NewNumericFeature(1, 1, 1, 1, 2, 2, 2, 2)

	continuousTarget := db.NewNumericFeature(3, 0, 3, 1, 7, 6, 5, -1)
	categoricalTarget := db.NewCategoricalFeature("1", "0", "1", "0", "1", "1", "1", "0")

	dtFeatures := []db.OrderedFeature{}
	for _, f := range []*db.NumericFeature{x, y, z} {
		dtFeatures = append(dtFeatures, f)
	}

	dt := db.NewDecisionTree()

	// Build a regression tree with the continuous target
	log.Printf("Regression tree; continuous target variables (really a regression)")
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

	// Grow a *regression* tree using categorical target but with
	// MSE criterion (as if it were a continuous target)
	log.Printf("Regression tree; categorical target")
	dt.Fit(dtFeatures, categoricalTarget, db.NewMSECriterionFactory())
	fmt.Printf("%v\n", dt.Importances())

	tEstimate = dt.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != categoricalTarget.Len() {
		test.Errorf("dt.Predict() returned result of length: %d; expected %d", len(tEstimate), categoricalTarget.Len())
	}

	for i, te := range tEstimate {
		if te != categoricalTarget.Value(i) {
			test.Errorf("Row %d: predicted %v; actual %v", i, te, categoricalTarget.Value(i))
		}
	}

	// Repeat using categorical target with Gini criterion
	log.Printf("Classification tree (gini); categorical target")
	dt.Fit(dtFeatures, categoricalTarget, db.NewGiniCriterionFactory(categoricalTarget.Len()))
	fmt.Printf("%v\n", dt.Importances())

	// Repeat using categoricalTarget and entropy criterion
	log.Printf("Classification tree (entropy); categorical target")
	dt.Fit(dtFeatures, categoricalTarget, db.NewEntropyCriterionFactory(categoricalTarget.Len()))
	fmt.Printf("%v\n", dt.Importances())

	tEstimate = dt.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != categoricalTarget.Len() {
		test.Errorf("dt.Predict() returned result of length: %d; expected %d", len(tEstimate), continuousTarget.Len())
	}

	for i, te := range tEstimate {
		if te != categoricalTarget.NumericValue(i) {
			test.Errorf("Row %d: predicted %v; actual %v", i, te, categoricalTarget.Value(i))
		}
	}
}
