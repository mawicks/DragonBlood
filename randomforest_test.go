package DragonBlood_test

import (
	"fmt"
	"testing"

	db "github.com/mawicks/DragonBlood"
)

func TestRandomForest(test *testing.T) {
	x := db.NewNumericFeature(0, 1, 2, 3, 4, 5, 6, 7)
	y := db.NewNumericFeature(1, 2, 1, 2, 1, 2, 1, 2)
	z := db.NewNumericFeature(1, 1, 1, 1, 2, 2, 2, 2)

	numerict := db.NewNumericFeature(0, 0, 0, 0, 1, 1, 1, 1)
	categoricalt := db.NewCategoricalFeature("0", "0", "0", "0", "1", "1", "1", "1")
	//	t.Add(3, 0, 3, 1, 7, 6, 5, -1)

	rfFeatures := []db.DTFeature{}
	for _, f := range []*db.NumericFeature{x, y, z} {
		rfFeatures = append(rfFeatures, db.NewDTNumericFeature(f))
	}

	rf := db.NewRandomForest(10).SetMaxFeatures(3).SetMinLeafSize(1)

	oob := rf.Fit(rfFeatures, db.NewMSECriterionTarget(numerict))

	tEstimate := rf.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != numerict.Len() {
		test.Errorf("rf.Predict() returned result of length: %d; expected %d", len(tEstimate), numerict.Len())
	}

	fmt.Printf("oob preds: %v\n", oob)

	for i, te := range tEstimate {
		fmt.Printf("predicted: %v; actual: %v\n", te, numerict.Value(i))
		//		if te != t.Value(i) {
		//			test.Errorf("Row %d: predicted %v; actual %v", i, te, numerict.Value///(i))
		//		}
	}

	fmt.Printf("feature importances (forest): %v\n", rf.Importances())

	oob = rf.Fit(rfFeatures, db.NewGiniCriterionTarget(categoricalt))

	tEstimate = rf.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != categoricalt.Len() {
		test.Errorf("rf.Predict() returned result of length: %d; expected %d", len(tEstimate), categoricalt.Len())
	}

	fmt.Printf("oob preds: %v\n", oob)

	for i, te := range tEstimate {
		fmt.Printf("predicted: %v; actual: %v\n", te, categoricalt.NumericValue(i))
		//		if te != t.Value(i) {
		//			test.Errorf("Row %d: predicted %v; actual %v", i, te, categoricalt.Value///(i))
		//		}
	}

	fmt.Printf("feature importances (forest): %v\n", rf.Importances())
}
