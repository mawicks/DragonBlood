package DragonBlood

import (
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"

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

type Metric struct {
	size       int
	prediction float64
	metric     float64
}

// DecisionTreeNodetype describes an arbitrary node in a decision tree.
type DecisionTreeNode struct {
	Metric

	// Left and Right children (nil values means this is a leaf)
	Left, Right *DecisionTreeNode

	// Feature index to split on.  The value -1 represents a leaf
	feature int

	// Splitter to use on above feature if feature>= 0 else nil.
	splitter Splitter
}

// Dump prints a readable representation of a decision tree
func (n *DecisionTreeNode) Dump(w io.Writer, level int, prefix string) {
	for i := 0; i < level; i++ {
		fmt.Fprint(w, " ")
	}
	fmt.Fprintf(w, "%sprediction: %g metric: %g size: %d", prefix, n.prediction, n.metric, n.size)
	if n.feature < 0 {
		fmt.Fprint(w, " (LEAF)")
	}
	fmt.Fprint(w, "\n")
	if n.Left != nil {
		n.Left.Dump(w, level+4, fmt.Sprintf("L feature_%d %s ", n.feature, n.splitter))
	}
	if n.Right != nil {
		n.Right.Dump(w, level+4, fmt.Sprintf("R feature_%d not %s ", n.feature, n.splitter))
	}
}

type SplitInfo struct {
	splitter Splitter
	metric   float64

	left  Metric
	right Metric
}

type FeatureSplitInfo struct {
	feature int
	*SplitInfo
}

func (fsi *FeatureSplitInfo) Dump() {
	if fsi != nil {
		log.Printf("\tfeature %v: %+v\n", fsi.feature, fsi.SplitInfo)
	} else {
		log.Printf("\tnil\n")
	}
}

type MSEAccumulator struct {
	minLeafSize          int
	previousFeatureValue float64
	left, right          *stats.SSEAccumulator

	count          int
	bestMetric     float64
	bestSplitValue float64

	bestLeft  Metric
	bestRight Metric
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
		metric := (a.left.Value() + a.right.Value())
		if metric < a.bestMetric && a.left.Count() >= a.minLeafSize && a.right.Count() >= a.minLeafSize {
			a.bestMetric = metric
			a.bestSplitValue = 0.5 * (featureValue + a.previousFeatureValue)

			if a.bestLeft.size = a.left.Count(); a.bestLeft.size > 0 {
				a.bestLeft.metric = a.left.MSE()
				a.bestLeft.prediction = a.left.Mean()
			}

			if a.bestRight.size = a.right.Count(); a.bestRight.size > 0 {
				a.bestRight.metric = a.right.MSE()
				a.bestRight.prediction = a.right.Mean()
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

	if a.bestLeft.size != 0 && a.bestRight.size != 0 {
		result = &SplitInfo{
			splitter: NumericSplitter(a.bestSplitValue),
			metric:   a.bestMetric,
			left:     a.bestLeft,
			right:    a.bestRight,
		}
	}
	return result
}

// dtOptimalSplit computes an optimal split for feature f using target.
// nodeMembership maps each training unit to the index of splittableNode to which it currently belongs.
// nodeCount is the number of splittableNodes (one more than the max value of nodeMembership)
// bag maps each unit to the number of times that unit occurs in the current bag.
// minSize is the minimum size for a leaf node.
func dtOptimalSplit(
	f DecisionTreeFeature,
	target DecisionTreeTarget,
	nodeMembership []int,
	nodeCount int,
	bag Bag,
	minSize int) []*SplitInfo {

	if minSize <= 0 {
		minSize = 1
	}

	accumulators := make([]*MSEAccumulator, nodeCount)
	for i, _ := range accumulators {
		accumulators[i] = NewMSEAccumulator(minSize)
	}

	// First pass - accumulate stats with all points to right of split
	// Add points from right to left so they are removed in LIFO order
	for i := len(nodeMembership) - 1; i >= 0; i-- {
		iOrdered := f.InOrder(i)
		if nm := nodeMembership[iOrdered]; nm >= 0 {
			for j := 0; j < bag.Count(iOrdered); j++ {
				accumulators[nm].Add(target.NumericValue(iOrdered))
			}
		}
	}

	// Second pass - move points from right to left and evalute new metric
	for i, _ := range nodeMembership {
		iOrdered := f.InOrder(i)
		if nm := nodeMembership[iOrdered]; nm >= 0 {
			for j := 0; j < bag.Count(iOrdered); j++ {
				accumulators[nm].Move(f.NumericValue(iOrdered), target.NumericValue(iOrdered))
			}
		}
	}

	// Collect results from accumulators
	result := make([]*SplitInfo, nodeCount)
	for i, _ := range result {
		if bestSplit := accumulators[i].BestSplit(); bestSplit != nil {
			result[i] = bestSplit
		} else {
			result[i] = nil
		}
	}

	return result
}

type DecisionTree struct {
	MaxFeatures int
	MinLeafSize int
}

func dtInitialize(target DecisionTreeTarget, bag Bag) ([]*DecisionTreeNode, []int) {
	node := &DecisionTreeNode{feature: -1}

	splittableNodeMembership := make([]int, bag.Len())
	acc := stats.NewSSEAccumulator()
	for i := 0; i < bag.Len(); i++ {
		splittableNodeMembership[i] = 0 // Root node
		for j := 0; j < bag.Count(i); j++ {
			acc.Add(target.NumericValue(i))
		}
	}

	node.Metric = Metric{size: acc.Count(), prediction: acc.Mean(), metric: acc.MSE()}

	// nextSplittableNodes is next generation of splittableNodes.
	// It is initialized here (and re-generated during each
	// iteration) and contains pointers to nodes that are eligible
	// for splitting during next generation.
	initialSplittableNodes := []*DecisionTreeNode{node}

	return initialSplittableNodes, splittableNodeMembership
}

func dtSelectSplits(splittableNodes []*DecisionTreeNode,
	candidateSplitsByFeature [][]*FeatureSplitInfo,
	maxFeatures int) ([]*DecisionTreeNode, []*SplitPair) {

	nextSplittableNodes := make([]*DecisionTreeNode, 0, 2*len(splittableNodes))
	// nodeSplits is generated during each iteration and
	// contains the next-generation indexes of children of
	// nodes to be split (left and right values of
	// SplitPair are indexes of nextSplittableNodes; index
	// of nodeSplits match those of splittableNodes))
	nodeSplits := make([]*SplitPair, 0, len(splittableNodes))

	improvingSplits := make([]*FeatureSplitInfo, 0, len(candidateSplitsByFeature))
	log.Print("Selected feature/split by eligible node: ")
	for inode, node := range splittableNodes {
		// For this node, build list of feature splits
		// that reduce the metric
		improvingSplits = improvingSplits[:0]
		for _, nodeCandidateSplits := range candidateSplitsByFeature {
			if nodeCandidateSplits[inode] != nil && nodeCandidateSplits[inode].metric < node.metric {
				improvingSplits = append(improvingSplits, nodeCandidateSplits[inode])
			}
		}

		// Consider a random subset of feature splits of size maxFeatures and pick the best of those
		var bestSplit *FeatureSplitInfo
		for i := 0; i < maxFeatures && i < len(improvingSplits); i++ {
			irand := i + rand.Intn(len(improvingSplits)-i)
			improvingSplits[i], improvingSplits[irand] = improvingSplits[irand], improvingSplits[i]
			if bestSplit == nil || improvingSplits[i].metric < bestSplit.metric {
				bestSplit = improvingSplits[i]
			}
		}

		var newPair *SplitPair = nil
		if bestSplit != nil {
			node.feature = bestSplit.feature
			node.splitter = bestSplit.splitter

			node.Left = &DecisionTreeNode{feature: -1, Metric: bestSplit.left}
			node.Right = &DecisionTreeNode{feature: -1, Metric: bestSplit.right}

			leftIndex := len(nextSplittableNodes)
			nextSplittableNodes = append(nextSplittableNodes, node.Left)

			rightIndex := len(nextSplittableNodes)
			nextSplittableNodes = append(nextSplittableNodes, node.Right)

			newPair = &SplitPair{left: leftIndex, right: rightIndex}
		}
		nodeSplits = append(nodeSplits, newPair)

		bestSplit.Dump()
	}
	return nextSplittableNodes, nodeSplits
}

type SplitPair struct{ left, right int }

func (dt *DecisionTree) Grow(features []DecisionTreeFeature, target DecisionTreeTarget, bag Bag) *DecisionTreeNode {
	maxFeatures := dt.MaxFeatures
	if maxFeatures > len(features) || maxFeatures <= 0 {
		maxFeatures = len(features)
	}

	initialSplittableNodes, splittableNodeMembership := dtInitialize(target, bag)
	root := initialSplittableNodes[0]

	// candidateSplitsByFeature is a fixed length slices that is
	// re-used during each iteration
	candidateSplitsByFeature := make([][]*FeatureSplitInfo, len(features))

	var nextSplittableNodes []*DecisionTreeNode
	for splittableNodes := initialSplittableNodes; len(splittableNodes) > 0; splittableNodes = nextSplittableNodes {
		log.Printf("*** New Iteration ***:  splittableNodeMembership: %v", splittableNodeMembership)

		// For each feature find all optimal splits for that feature for each splittable node
		for i, feature := range features {
			candidateSplitsByFeature[i] = make([]*FeatureSplitInfo, 0, len(splittableNodes))
			log.Printf("Best splits by node, feature %d:\n", i)
			for _, dtos := range dtOptimalSplit(feature, target, splittableNodeMembership, len(splittableNodes), bag, dt.MinLeafSize) {
				var split *FeatureSplitInfo
				if dtos != nil {
					split = &FeatureSplitInfo{i, dtos}
				}
				candidateSplitsByFeature[i] = append(candidateSplitsByFeature[i], split)
				split.Dump()
			}
		}

		var nodeSplits []*SplitPair
		nextSplittableNodes, nodeSplits = dtSelectSplits(splittableNodes, candidateSplitsByFeature, maxFeatures)

		// Update node membership
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
					if bag.Count(i) > 0 {
						// TODO:  Compute OOB scores
					}
				}
			}
		}
	}
	log.Print("Tree Dump:")
	root.Dump(os.Stderr, 0, "Root ")
	return root
}

func (dt *DecisionTreeNode) Predict(features []Feature) []float64 {
	var result []float64
	if len(features) > 0 {
		result = make([]float64, features[0].Len())
		for i := range result {
			node := dt
			for node.feature >= 0 {
				if node.splitter.Split(features[node.feature].NumericValue(i)) {
					node = node.Left
				} else {
					node = node.Right
				}
			}
			result[i] = node.prediction
		}
	}
	return result
}

type randomForestNumericFeature struct {
	*NumericFeature
}

func NewDecisionTreeNumericFeature(f *NumericFeature) DecisionTreeFeature {
	return &randomForestNumericFeature{
		f,
	}
}

type DecisionTreeRegressor struct {
	root *DecisionTreeNode
}

func NewDecisionTreeRegressor() *DecisionTreeRegressor {
	return &DecisionTreeRegressor{}
}

func (dtr *DecisionTreeRegressor) Fit(features []DecisionTreeFeature, target DecisionTreeTarget) {
	//	bag := NewBag(features[0].Len())
	bag := FullBag(features[0].Len())
	log.Printf("bag: %v", bag)
	for _, f := range features {
		f.Sort()
	}

	dt := &DecisionTree{MaxFeatures: 10, MinLeafSize: 1}
	dtr.root = dt.Grow(features, target, bag)
}

func (dtr *DecisionTreeRegressor) Predict(features []Feature) []float64 {
	var result []float64
	if dtr.root != nil {
		result = dtr.root.Predict(features)
	}
	return result
}
