package DragonBlood

import (
	"fmt"
	"log"
	"math"
)

type RandomForestRegressor struct {
	nTrees    int
	trees     []*DTNode
	nFeatures int
	grower    *decisionTreeGrower
}

func NewRandomForest(nTrees int) *RandomForestRegressor {
	return &RandomForestRegressor{
		nTrees,
		make([]*DTNode, 0, nTrees),
		0,
		&decisionTreeGrower{MaxFeatures: math.MaxInt32, MinLeafSize: 1},
	}
}

// Parameter setters:
func (rf *RandomForestRegressor) SetMinLeafSize(m int) *RandomForestRegressor {
	rf.grower.MinLeafSize = m
	return rf
}

func (rf *RandomForestRegressor) SetMaxFeatures(m int) *RandomForestRegressor {
	rf.grower.MaxFeatures = m
	return rf
}

func (rf *RandomForestRegressor) Fit(features []DTFeature, target DTTarget, af DTSplittingCriterionFactory) []float64 {
	rf.nFeatures = len(features)

	oobPrediction := make([]DTSplittingCriterion, features[0].Len())
	for i := range oobPrediction {
		oobPrediction[i] = af.New()
	}

	for i := 0; i < rf.nTrees; i++ {
		bag := NewBag(features[0].Len())
		log.Printf("Next Fit: %d\n", i)
		//		log.Printf("   --- bag: %v", bag)

		rf.trees = append(rf.trees, rf.grower.grow(features, target, bag, oobPrediction, af))
	}

	result := make([]float64, len(oobPrediction))
	for i, p := range oobPrediction {
		result[i] = p.Prediction()
	}

	return result
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

func (rf *RandomForestRegressor) Importances() []float64 {
	fmt.Printf("Importances(): nFeatures: %d\n", rf.nFeatures)

	forestImportances := make([]float64, rf.nFeatures)
	treeImportances := make([]float64, rf.nFeatures)

	for i, tree := range rf.trees {
		if tree != nil {
			for j := range treeImportances {
				treeImportances[j] = 0.0
			}

			tree.Importances(treeImportances)
			for j, imp := range treeImportances {
				forestImportances[j] += (imp - forestImportances[j]) / float64(i+1)
			}
		}
	}
	return forestImportances
}
