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

type Splitter interface {
	Split(float64) bool
	String() string
}

// Types

// NumericSplitter is an implementation of Splitter that performs a simple split of an ordered feature variable.
type NumericSplitter float64

func (s NumericSplitter) Split(x float64) bool { return x < float64(s) }
func (s NumericSplitter) String() string       { return fmt.Sprintf("< %g", float64(s)) }

// DecisionTreeNodetype describes an arbitrary node in a decision tree.
type DecisionTreeNode struct {
	size       int
	prediction float64
	metric     float64

	// Left and Right children (nil values means this is a leaf)
	Left, Right *DecisionTreeNode

	// Feature index to split on.  The value -1 represents a leaf
	feature int

	// Splitter to use on above feature if feature>= 0 else nil.
	splitter Splitter
}

// Dump prints a readable representation of a decision tree
func (n *DecisionTreeNode) Dump(level int, prefix string) {
	for i := 0; i < level; i++ {
		fmt.Print(" ")
	}
	fmt.Printf("%sprediction: %g", prefix, n.prediction)
	if n.feature < 0 {
		fmt.Printf(" (LEAF)")
	}
	fmt.Println()
	if n.Left != nil {
		n.Left.Dump(level+4, fmt.Sprintf("L feature_%d %s ", n.feature, n.splitter))
	}
	if n.Right != nil {
		n.Right.Dump(level+4, fmt.Sprintf("R feature_%d not %s ", n.feature, n.splitter))
	}
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
	minLeafSize          int
	previousFeatureValue float64
	left, right          *stats.SSEAccumulator

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

func NewMSEAccumulator(minLeafSize int) *MSEAccumulator {
	return &MSEAccumulator{
		left:                 stats.NewSSEAccumulator(),
		right:                stats.NewSSEAccumulator(),
		bestMetric:           math.Inf(1),
		previousFeatureValue: math.Inf(-1),
		count:                0,
		minLeafSize:          minLeafSize,
	}
}

// Add target value to "right" tally
func (a *MSEAccumulator) Add(targetValue float64) {
	a.right.Add(targetValue)
	a.count += 1
}

// Evaluate the metric assuming a break to the immediate left of
// featureValue (feature Value is in right set).  Then move the
// attribute value from the right set to left set.
func (a *MSEAccumulator) Move(featureValue, targetValue float64) {
	if featureValue != a.previousFeatureValue { // End of a run of identical values
		metric := (a.left.Value() + a.right.Value()) / float64(a.count)
		if metric < a.bestMetric {
			a.bestMetric = metric
			a.bestSplitValue = 0.5 * (featureValue + a.previousFeatureValue)

			if a.bestLeftSize = a.left.Count(); a.bestLeftSize > 0 {
				a.bestLeftMetric = a.left.MSE()
				a.bestLeftPrediction = a.left.Mean()
			}

			if a.bestRightSize = a.right.Count(); a.bestRightSize > 0 {
				a.bestRightMetric = a.right.MSE()
				a.bestRightPrediction = a.right.Mean()
			}
		}
		a.previousFeatureValue = featureValue
	}

	a.right.Subtract(targetValue)
	a.left.Add(targetValue)
}

func (a *MSEAccumulator) BestSplit() *SplitInfo {
	var result *SplitInfo

	if a.right.Count() != 0 {
		panic("BestSplit() called prematurely (fewer Move() calls than Add() calls)")
	}

	if a.bestLeftSize != 0 && a.bestRightSize != 0 {
		result = &SplitInfo{
			splitter:        NumericSplitter(a.bestSplitValue),
			metric:          a.bestMetric,
			leftSize:        a.bestLeftSize,
			rightSize:       a.bestRightSize,
			leftPrediction:  a.bestLeftPrediction,
			rightPrediction: a.bestRightPrediction,
			leftMetric:      a.bestLeftMetric,
			rightMetric:     a.bestRightMetric,
		}
	}

	return result
}

// dtOptimalSplit computes an optimal split for feature f using target.
// It evalutes every node in splittableNodes for a possible split.
// nodeMembership maps each training unit to the index of splittableNode to which it currently belongs.
// bag maps each unit to the number of times that unit occurs in the current bag.
// minSize is the minimum size for a leaf node.
func dtOptimalSplit(
	f DecisionTreeFeature,
	target DecisionTreeTarget,
	splittableNodes []*DecisionTreeNode,
	nodeMembership []int,
	bag Bag,
	minSize int) []*SplitInfo {

	fmt.Printf("\ndtOptimalSplit:\n\tfeature %+v\n\ttarget: %+v\n\tbag: %+v\n", f, target, bag)

	accumulators := make([]*MSEAccumulator, len(splittableNodes))
	for i, _ := range accumulators {
		accumulators[i] = NewMSEAccumulator(minSize)
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
	for i, _ := range nodeMembership {
		iOrdered := f.InOrder(i)
		if nm := nodeMembership[iOrdered]; nm >= 0 {
			for j := 0; j < bag[iOrdered]; j++ {
				accumulators[nm].Move(f.NumericValue(iOrdered), target.NumericValue(iOrdered))
			}
		}
	}

	result := make([]*SplitInfo, len(splittableNodes))
	for i, _ := range result {
		if bestSplit := accumulators[i].BestSplit(); bestSplit != nil {
			result[i] = bestSplit
		} else {
			result[i] = nil
		}
	}

	fmt.Printf("dtOptimalSplit returning:\n")
	for _, os := range result {
		if os != nil {
			fmt.Printf("\t%+v\n", *os)
		} else {
			fmt.Printf("nil\n")
		}
	}
	return result
}

type DecisionTree struct {
	maxFeatures int
	minLeafSize int
}

func dtInitialize(target DecisionTreeTarget, bag Bag) ([]*DecisionTreeNode, []int) {
	node := &DecisionTreeNode{feature: -1}

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

func (dt *DecisionTree) Fit(features []DecisionTreeFeature, target DecisionTreeTarget, bag Bag) *DecisionTreeNode {
	maxFeatures := dt.maxFeatures
	if dt.maxFeatures > len(features) {
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
		// nodes to be split (left and right values of
		// SplitPair are indexes of nextSplittableNodes; index
		// of nodeSplits match those of splittableNodes))
		nodeSplits := make([]*SplitPair, 0, len(splittableNodes))
		bestSplits := make([]*FeatureSplitInfo, 0, len(splittableNodes))

		fmt.Println("\n\n*** New Iteration ***")
		fmt.Printf("splittableNodeMembership: %v\n", splittableNodeMembership)

		for i, feature := range features {
			candidateSplitsByFeature[i] = make([]*FeatureSplitInfo, 0, len(splittableNodes))
			for _, dtos := range dtOptimalSplit(feature, target, splittableNodes, splittableNodeMembership, bag, dt.minLeafSize) {
				var split *FeatureSplitInfo
				if dtos != nil {
					split = &FeatureSplitInfo{i, dtos}
				}
				candidateSplitsByFeature[i] = append(candidateSplitsByFeature[i], split)
			}
		}

		candidateSplits := make([]*FeatureSplitInfo, 0, len(features))
		for inode, node := range splittableNodes {
			candidateSplits = candidateSplits[:0]
			for _, nodeCandidateSplits := range candidateSplitsByFeature {
				if nodeCandidateSplits[inode] != nil && nodeCandidateSplits[inode].metric < node.metric {
					candidateSplits = append(candidateSplits, nodeCandidateSplits[inode])
				}
			}

			var bestSplit *FeatureSplitInfo
			for i := 0; i < maxFeatures && i < len(candidateSplits); i++ {
				irand := i + rand.Intn(len(candidateSplits)-i)
				candidateSplits[i], candidateSplits[irand] = candidateSplits[irand], candidateSplits[i]
				if bestSplit == nil || candidateSplits[i].metric < bestSplit.metric {
					bestSplit = candidateSplits[i]
				}
			}
			var newPair *SplitPair = nil
			if bestSplit != nil {
				node.Left = &DecisionTreeNode{feature: -1, size: bestSplit.leftSize, prediction: bestSplit.leftPrediction, metric: bestSplit.leftMetric}
				node.Right = &DecisionTreeNode{feature: -1, size: bestSplit.rightSize, prediction: bestSplit.rightPrediction, metric: bestSplit.rightMetric}

				node.feature = bestSplit.feature
				node.splitter = bestSplit.splitter

				leftIndex := len(nextSplittableNodes)
				nextSplittableNodes = append(nextSplittableNodes, node.Left)

				rightIndex := len(nextSplittableNodes)
				nextSplittableNodes = append(nextSplittableNodes, node.Right)

				fmt.Printf("leftIndex: %d rightIndex: %d\n", leftIndex, rightIndex)
				newPair = &SplitPair{left: leftIndex, right: rightIndex}
			}
			nodeSplits = append(nodeSplits, newPair)
			bestSplits = append(bestSplits, bestSplit)
		}
		fmt.Println("Best splits: ")
		for _, bs := range bestSplits {
			if bs != nil {
				fmt.Printf("feature: %v %+v\n", bs.feature, bs.SplitInfo)
			} else {
				fmt.Printf("feature: nil\n")
			}
		}

		// Assign each unit to its child node
		for i, sn := range splittableNodeMembership {
			if sn >= 0 {
				splittableNode := splittableNodes[sn]
				if splittableNode.feature >= 0 {
					if splittableNode.splitter.Split(features[splittableNode.feature].NumericValue(i)) { // Left
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
		fmt.Printf("splittableNodeMembership: %v\n", splittableNodeMembership)
		fmt.Printf("nextSplittableNodes: %v\n", nextSplittableNodes)
		fmt.Printf("**** LOOPING ****\n")
	}
	fmt.Printf("\nTree Dump:\n")
	root.Dump(0, "Root ")
	fmt.Printf("\n\n")
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

	for _, f := range features {
		f.Sort()
	}

	dt := &DecisionTree{10, 1}
	dt.Fit(features, target, bag)

}

func (rf *DecisionTreeRegressor) Predict(features []Feature) []float64 {
	return make([]float64, 0)
}
