package DragonBlood_test

import (
	"fmt"
	"testing"

	db "github.com/mawicks/DragonBlood"
)

func TestRandomForest(test *testing.T) {
	x := db.NewNumericFeature(nil)
	x.Add(0, 1, 2, 3, 4, 5, 6, 7)

	y := db.NewNumericFeature(nil)
	y.Add(1, 2, 1, 2, 1, 2, 1, 2)

	z := db.NewNumericFeature(nil)
	z.Add(1, 1, 1, 1, 2, 2, 2, 2)

	t := db.NewNumericFeature(nil)
	//	t.Add(3, 0, 3, 1, 7, 6, 5, -1)
	t.Add(0, 0, 0, 0, 1, 1, 1, 1)

	rfFeatures := []db.OrderedFeature{}
	for _, f := range []*db.NumericFeature{x, y, z} {
		rfFeatures = append(rfFeatures, f)
	}

	rf := db.NewRandomForestRegressor(10).SetMaxFeatures(3).SetMinLeafSize(1)
	oob := rf.Fit(rfFeatures, t)

	tEstimate := rf.Predict([]db.Feature{x, y, z})

	if len(tEstimate) != t.Len() {
		test.Errorf("rf.Predict() returned result of length: %d; expected %d", len(tEstimate), t.Len())
	}

	fmt.Printf("oob preds: %v\n", oob)

	for i, te := range tEstimate {
		fmt.Printf("predicted: %v; actual: %v\n", te, t.Value(i))
		//		if te != t.Value(i) {
		//			test.Errorf("Row %d: predicted %v; actual %v", i, te, t.Value///(i))
		//		}
	}

	fmt.Printf("feature importances (forest): %v\n", rf.Importances())
}
