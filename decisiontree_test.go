package DragonBlood_test

import (
	"fmt"
	"log"
	"testing"

	db "github.com/mawicks/DragonBlood"
)

func TestDecisionTree(test *testing.T) {
	x := db.NewCategoricalFeature(0, 1, 2, 3, 4, 5, 6, 7)
	y := db.NewCategoricalFeature(1, 2, 1, 2, 1, 2, 1, 2)
	z := db.NewCategoricalFeature(1, 1, 1, 1, 2, 2, 2, 2)

	continuousTarget := db.NewNumericFeature(3, 0, 3, 1, 7, 6, 5, -1)
	categoricalTarget := db.NewCategoricalFeature(1, 0, 1, 0, 1, 1, 1, 0)
	binaryContinuousTarget := db.NewNumericFeature(1, 0, 1, 0, 1, 1, 1, 0)

	categoricalTarget.OrderEncoding()

	dtFeatures := []db.DTFeature{}
	for _, f := range []*db.CategoricalFeature{x, y, z} {
		dtFeatures = append(dtFeatures, db.NewDTCategoricalFeature(f))
	}

	dt := db.NewDecisionTree()

	// Build a regression tree with the continuous target
	log.Printf("Regression tree; continuous target variables (really a regression)")
	dt.Fit(dtFeatures, db.NewMSECriterionTarget(continuousTarget))
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

	// Grow a *regression* tree using "categorical" target (ones
	// and zeros in a continuous target) with MSE criterion.
	log.Printf("Regression tree; categorical target")
	dt.Fit(dtFeatures, db.NewMSECriterionTarget(binaryContinuousTarget))
	fmt.Printf("%v\n", dt.Importances())

	tEstimate = dt.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != categoricalTarget.Len() {
		test.Errorf("dt.Predict() returned result of length: %d; expected %d", len(tEstimate), categoricalTarget.Len())
	}

	for i, te := range tEstimate {
		if te != categoricalTarget.NumericValue(i) {
			test.Errorf("Row %d: predicted %v; actual %v", i, te, categoricalTarget.Value(i))
		}
	}

	// Repeat using categorical target with Gini criterion
	log.Printf("Classification tree (gini); categorical target")
	dt.Fit(dtFeatures, db.NewGiniCriterionTarget(categoricalTarget))
	fmt.Printf("%v\n", dt.Importances())

	// Repeat using categoricalTarget and entropy criterion
	log.Printf("Classification tree (entropy); categorical target")
	dt.Fit(dtFeatures, db.NewEntropyCriterionTarget(categoricalTarget))
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
