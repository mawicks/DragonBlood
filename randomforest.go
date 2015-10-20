package DragonBlood

import (
	"fmt"
	"log"

	"github.com/mawicks/DragonBlood/stats"
)

type RandomForestRegressor struct {
	nTrees    int
	trees     []*DecisionTreeNode
	nFeatures int
	grower    *decisionTreeGrower
}

func NewRandomForestRegressor(nTrees int) *RandomForestRegressor {
	return &RandomForestRegressor{
		nTrees,
		make([]*DecisionTreeNode, 0, nTrees),
		0,
		&decisionTreeGrower{MaxFeatures: 10, MinLeafSize: 1},
	}
}

func (rf *RandomForestRegressor) Fit(features []DecisionTreeFeature, target DecisionTreeTarget) []float64 {
	rf.nFeatures = len(features)

	oobPrediction := make([]stats.Accumulator, features[0].Len())
	for i := range oobPrediction {
		oobPrediction[i] = stats.NewMeanAccumulator()
	}

	for _, f := range features {
		f.Sort()
	}

	for i := 0; i < rf.nTrees; i++ {
		bag := NewBag(features[0].Len())
		log.Printf("bag: %v", bag)

		rf.trees = append(rf.trees, rf.grower.grow(features, target, bag, oobPrediction))
	}

	result := make([]float64, len(oobPrediction))
	for i, p := range oobPrediction {
		result[i] = p.Value()
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
