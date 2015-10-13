package DragonBlood

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/mawicks/DragonBlood/stats"
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
	size       int
	prediction float64
	metric     float64

	// Left and Right children (nil values means this is a leaf)
	Left, Right *DecisionTreeNode

	// Feature index to split on
	feature int // Undefined in a leaf
	// Splitter to use on above feature
	splitter Splitter // Also nil in a leaf
}

type SplitInfo struct {
	splitter Splitter
	metric   float64

	leftSize, rightSize             int
	leftPrediction, rightPrediction float64
	leftMetric, rightMetric         float64
}

type FeatureSplitInfo struct {
	feature int
	*SplitInfo
}

type MSEAccumulator struct {
	previousValue float64
	left, right   *stats.SSEAccumulator

	count          int
	bestMetric     float64
	bestSplitValue float64

	bestLeftSize       int
	bestLeftMetric     float64
	bestLeftPrediction float64

	bestRightSize       int
	bestRightMetric     float64
	bestRightPrediction float64
}

func NewMSEAccumulator() *MSEAccumulator {
	return &MSEAccumulator{
		left:          stats.NewSSEAccumulator(),
		right:         stats.NewSSEAccumulator(),
		bestMetric:    math.Inf(1),
		previousValue: math.Inf(-1),
	}
}

// Add to "right" tally
func (a *MSEAccumulator) Add(t float64) {
	a.right.Add(t)
	a.count += 1
}

// Move an attribute value from right to left. Return resulting
// composite metric for this split.
func (a *MSEAccumulator) Move(f, t float64) {
	fmt.Printf("Move(%v, %v)\n", f, t)
	fmt.Printf("\t before left value: %v; ", a.left.Value())
	fmt.Printf("right value: %v\n", a.right.Value())
	fmt.Printf("\t before left mean: %v; ", a.left.Mean())
	fmt.Printf("right mean: %v\n", a.right.Mean())

	if f != a.previousValue { // End of a run of identical values
		metric := (a.left.Value() + a.right.Value()) / float64(a.count)
		if metric < a.bestMetric {
			a.bestMetric = metric
			a.bestSplitValue = 0.5 * (f + a.previousValue)

			if a.bestLeftSize = a.left.Count(); a.bestLeftSize > 0 {
				a.bestLeftMetric = a.left.MSE()
				a.bestLeftPrediction = a.left.Mean()
			}

			if a.bestRightSize = a.right.Count(); a.bestRightSize > 0 {
				a.bestRightMetric = a.right.MSE()
				a.bestRightPrediction = a.right.Mean()
			}
		}
		a.previousValue = f
	}
	a.right.Subtract(t)
	a.left.Add(t)
	fmt.Printf("\t after left value: %v; ", a.left.Value())
	fmt.Printf("right value: %v\n", a.right.Value())
}

func (a *MSEAccumulator) BestSplit() *SplitInfo {
	var result *SplitInfo

	bestSplitValue := a.bestSplitValue
	if a.bestLeftSize != 0 && a.bestRightSize != 0 {
		result = &SplitInfo{
			// Closure is on copy of a.bestSplitValue, which can't change.
			splitter:        func(x float64) bool { return x < bestSplitValue },
			metric:          a.bestMetric,
			leftSize:        a.bestLeftSize,
			rightSize:       a.bestRightSize,
			leftPrediction:  a.bestLeftPrediction,
			rightPrediction: a.bestRightPrediction,
			leftMetric:      a.bestLeftMetric,
			rightMetric:     a.bestRightMetric,
		}
	} else {
		result = nil
	}
	if result != nil {
		fmt.Printf("returning: %+v", *result)
	} else {
		fmt.Printf("returning: nil")
	}

	return result
}

func dtOptimalSplit(
	iFeature int,
	f DecisionTreeFeature,
	target DecisionTreeTarget,
	splittableNodes []*DecisionTreeNode,
	nodeMembership []int,
	bag Bag,
	minSize int) []*FeatureSplitInfo {

	fmt.Printf("\ndtOptimalSplit:\n\tfeature: %+v\n\ttarget: %+v\n\tbag: %+v\n", f, target, bag)

	accumulators := make([]*MSEAccumulator, len(splittableNodes))
	for i, _ := range accumulators {
		accumulators[i] = NewMSEAccumulator()
	}

	// First pass - accumulate stats with all points to right of split
	// Add points from right to left so they are removed in LIFO order
	for i := len(nodeMembership) - 1; i >= 0; i-- {
		iOrdered := f.InOrder(i)
		if nm := nodeMembership[iOrdered]; nm >= 0 {
			for j := 0; j < bag[iOrdered]; j++ {
				accumulators[nm].Add(target.NumericValue(iOrdered))
			}
		}
	}

	// Second pass - move points from right to left and evalute new metric
	for i, nm := range nodeMembership {
		iOrdered := f.InOrder(i)
		if nm >= 0 {
			for j := 0; j < bag[iOrdered]; j++ {
				accumulators[nm].Move(f.NumericValue(iOrdered), target.NumericValue(iOrdered))
			}
		}
	}

	result := make([]*FeatureSplitInfo, len(splittableNodes))
	for i, _ := range result {
		result[i] = &FeatureSplitInfo{iFeature, accumulators[i].BestSplit()}
	}

	fmt.Printf("\treturning: %+v\n", result)
	return result
}

type DecisionTree struct{}

func dtInitialize(target DecisionTreeTarget, bag Bag) ([]*DecisionTreeNode, []int) {
	node := &DecisionTreeNode{}

	splittableNodeMembership := make([]int, len(bag))
	acc := stats.NewSSEAccumulator()
	for i, b := range bag {
		splittableNodeMembership[i] = 0 // Root node
		for j := 0; j < b; j++ {
			acc.Add(target.NumericValue(i))
		}
	}

	node.prediction = acc.Mean()
	node.metric = acc.MSE()

	// nextSplittableNodes is next generation of splittableNodes.
	// It is initialized here (and re-generated during each
	// iteration) and contains pointers to nodes that are eligible
	// for splitting during next generation.
	initialSplittableNodes := []*DecisionTreeNode{node}

	return initialSplittableNodes, splittableNodeMembership
}

func (dt *DecisionTree) Fit(features []DecisionTreeFeature, target DecisionTreeTarget, bag Bag, maxFeatures int, minSize int) *DecisionTreeNode {
	if maxFeatures > len(features) {
		maxFeatures = len(features)
	}

	initialSplittableNodes, splittableNodeMembership := dtInitialize(target, bag)
	root := initialSplittableNodes[0]

	type SplitPair struct{ left, right int }

	// candidateSplittsByFeature is a fixed length slices that is
	// re-used during each iteration
	candidateSplitsByFeature := make([][]*FeatureSplitInfo, len(features))

	var nextSplittableNodes []*DecisionTreeNode
	for splittableNodes := initialSplittableNodes; len(splittableNodes) > 0; splittableNodes = nextSplittableNodes {
		nextSplittableNodes = make([]*DecisionTreeNode, 0, 2*len(splittableNodes))

		// nodeSplits is generated during each iteration and
		// contains the next-generation indexes of children of
		// nodes that get split (left and right values of
		// SplitPair are indexes of nextSplittableNodes; index
		// of nodeSplits match those of splittableNodes))
		nodeSplits := make([]*SplitPair, 0, len(splittableNodes))

		for i, feature := range features {
			candidateSplitsByFeature[i] = dtOptimalSplit(i, feature, target, splittableNodes, splittableNodeMembership, bag, minSize)
		}
		candidateSplits := make([]*FeatureSplitInfo, 0, len(features))
		bestSplits := make([]*FeatureSplitInfo, 0, len(splittableNodes))

		for inode, node := range splittableNodes {
			candidateSplits = candidateSplits[:0]
			for _, nodeCandidateSplits := range candidateSplitsByFeature {
				if nodeCandidateSplits[inode].metric < node.metric {
					candidateSplits = append(candidateSplits, nodeCandidateSplits[inode])
				}
			}

			var bestSplit *FeatureSplitInfo
			for i := 0; i < maxFeatures && i < len(candidateSplits); i++ {
				irand := i + rand.Intn(len(features)-i)
				candidateSplits[i], candidateSplits[irand] = candidateSplits[irand], candidateSplits[i]
				if bestSplit == nil || candidateSplits[i].metric < bestSplit.metric {
					bestSplit = candidateSplits[i]
				}
			}
			if bestSplit != nil {
				node.splitter = bestSplit.splitter
				node.feature = bestSplit.feature

				leftChild := &DecisionTreeNode{size: bestSplit.leftSize, prediction: bestSplit.leftPrediction, metric: bestSplit.leftMetric}
				rightChild := &DecisionTreeNode{size: bestSplit.rightSize, prediction: bestSplit.rightPrediction, metric: bestSplit.rightMetric}

				leftIndex := len(nextSplittableNodes)
				nextSplittableNodes = append(nextSplittableNodes, leftChild)

				rightIndex := len(nextSplittableNodes)
				nextSplittableNodes = append(nextSplittableNodes, rightChild)

				nodeSplits = append(nodeSplits, &SplitPair{left: leftIndex, right: rightIndex})
			}
			bestSplits = append(bestSplits, bestSplit)
		}

		// Assign each unit to its child node
		for i, sn := range splittableNodeMembership {
			if sn >= 0 {
				splittableNode := splittableNodes[sn]
				if splittableNode.splitter != nil {
					if splittableNode.splitter(features[splittableNode.feature].NumericValue(i)) { // Left
						splittableNodeMembership[i] = nodeSplits[sn].left
					} else { // Right
						splittableNodeMembership[i] = nodeSplits[sn].right
					}
				} else { // No split exists --- this record has reached a leaf node.
					splittableNodeMembership[i] = -1 // An impossible node reference
					if bag[i] > 0 {
						// TODO:  Compute OOB scores
					}
				}
			}
		}
	}
	return root
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
	bag.Resample()
	fmt.Printf("%+v\n", bag)

	for _, f := range features {
		f.Sort()
		fmt.Printf("%+v\n", f)
	}
	dt := &DecisionTree{}
	dt.Fit(features, target, bag, 10, 1)

}

func (rf *DecisionTreeRegressor) Predict(features []Feature) []float64 {
	return make([]float64, 0)
}
