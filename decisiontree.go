package DragonBlood

import (
	"fmt"
	"math/rand"
)

// Interfaces
type DecisionTreeFeature interface {
	OrderedFeature
}

type DecisionTreeTarget interface {
	Feature
}

type DecisionTreeOrderedFeature interface {
	Feature
}

// Types

type Splitter func(float64) bool

type DecisionTreeNode struct {
	// Left and Right children (nil values means this is a leaf)
	Left  *DecisionTreeNode
	Right *DecisionTreeNode

	// Feature index to split on
	feature  int
	splitter Splitter

	predictions float64
	metric      float64
}

type SplitInfo struct {
	splitter Splitter
	metric   float64
}

type MSEAccumulator struct {
	leftMean, rightMean   float64
	leftcount, rightCount int
}

// Add to "right" tally
func (a *MSEAccumulator) Add(t float64, count int) {
	a.rightMean += float64(count) * (t - a.rightMean) / float64(a.rightCount+count)
	a.rightCount += count
}

// Move an  from right to left
func (a *MSEAccumulator) Move(t float64, count int) {

}

func optimalSplit(f DecisionTreeFeature, target DecisionTreeTarget, nodeMembership []*DecisionTreeNode, splittableNodes []*DecisionTreeNode, bag Bag) []SplitInfo {
	result := make([]SplitInfo, len(splittableNodes))
	return result
}

func (node *DecisionTreeNode) Fit(features []DecisionTreeFeature, target DecisionTreeTarget, bag Bag) {
	// All in-bag records initially belong to the root node
	nodeMembership := make([]*DecisionTreeNode, len(bag))
	for i, b := range bag {
		if b > 0 {
			nodeMembership[i] = node
		}
	}

	// Root node is initially the only node and it's splittable
	splittableNodes := []*DecisionTreeNode{node}

	candidateSplitsByFeature := make([][]SplitInfo, len(features))

	for len(splittableNodes) > 0 {
		for i, feature := range features {
			candidateSplitsByFeature[i] = optimalSplit(feature, target, nodeMembership, splittableNodes, bag)
		}
		for _, _ = range splittableNodes {
			// Need to sort features by node
		}
	}
}

// Bag
type Bag []int

func (b Bag) Resample() {
	for i, _ := range b {
		b[i] = 0
	}

	n := len(b)
	for i := 0; i < n; i++ {
		b[rand.Intn(n)] += 1
	}
}

func (b Bag) Len() int { return len(b) }

type randomForestNumericFeature struct {
	*NumericFeature
}

func NewDecisionTreeNumericFeature(f *NumericFeature) DecisionTreeFeature {
	return &randomForestNumericFeature{
		f,
	}
}

type DecisionTreeRegressor struct{}

func NewDecisionTreeRegressor() *DecisionTreeRegressor {
	return &DecisionTreeRegressor{}
}

func (rf *DecisionTreeRegressor) Fit(features []DecisionTreeFeature, target DecisionTreeTarget) {
	bag := Bag(make([]int, features[0].Len()))
	for i := 0; i < 3; i++ {
		bag.Resample()
		fmt.Printf("%#v\n", bag)
	}

	for _, f := range features {
		f.Sort()
		fmt.Printf("%#v\n", f)
	}
}

func (rf *DecisionTreeRegressor) Predict(features []Feature) []float64 {
	return make([]float64, 0)
}
