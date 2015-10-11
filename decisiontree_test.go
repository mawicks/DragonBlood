package DragonBlood_test

import (
	"testing"

	db "github.com/mawicks/DragonBlood"
)

func TestDecisionTree(test *testing.T) {
	x := db.NewNumericFeature("x")
	x.Add(0, 1, 2, 3, 4)

	y := db.NewNumericFeature("y")
	y.Add(1, 1, 1, 2, 2)

	z := db.NewNumericFeature("z")
	z.Add(1, 2, 1, 2, 1)

	t := db.NewNumericFeature("t")
	t.Add(3, 0, 3, 1, 7)

	rfFeatures := make([]db.DecisionTreeFeature, 0)
	for _, f := range []*db.NumericFeature{x, y, z} {
		rfFeatures = append(rfFeatures, db.NewDecisionTreeNumericFeature(f))
	}

	rf := db.NewDecisionTreeRegressor()
	rf.Fit(rfFeatures, t)

	tEstimate := rf.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != t.Len() {
		//		test.Errorf("rf.Predict() returned result of length: %d; expected %d", len(tEstimate), t.Len())
	}

	for i, te := range tEstimate {
		if te != t.Value(i) {
			test.Errorf("Row %d: predicted %v; actual %v", i, te, t.Value(i))
		}
	}
}
