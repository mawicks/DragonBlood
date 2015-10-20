package DragonBlood

import (
	"log"
)

type RandomForestRegressor struct {
	nTrees int
	trees  []*DecisionTree
}

func NewRandomForestRegressor(nTrees int) *RandomForestRegressor {
	return &RandomForestRegressor{nTrees, make([]*DecisionTree, 0, nTrees)}
}

func (rf *RandomForestRegressor) Fit(features []DecisionTreeFeature, target DecisionTreeTarget) {
	for _, f := range features {
		f.Sort()
	}

	dtg := &DecisionTreeGrower{MaxFeatures: 10, MinLeafSize: 1}

	for i := 0; i < rf.nTrees; i++ {
		bag := NewBag(features[0].Len())
		log.Printf("bag: %v", bag)

		rf.trees = append(rf.trees, dtg.Grow(features, target, bag))
	}
}

func (rf *RandomForestRegressor) Predict(features []Feature) []float64 {
	result := make([]float64, features[0].Len())

	for i, tree := range rf.trees {
		if tree != nil {
			for j, p := range tree.Predict(features) {
				result[j] += (p - result[j]) / float64(i+1)
			}
		}
	}

	return result
}
