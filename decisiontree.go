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
type Splitter interface {
	Split(float64) bool
	String() string
}

type DecisionTreeMetric interface {
	// Add is called once during a first pass on each member of a node to be split
	// to accumulate statistics for the case that all members are in the right node.
	Add(DecisionTreePredictor)

	// Move is called once on each member to evalute the change in metric
	// associated with moving it from the right node to the left node
	Move(featureValue, targetValue float64)

	// Bestsplit returns the best split encountered during the second pass.
	// The underlying acculator will typically accumulate both the prediction and the metric.
	BestSplit() *SplitInfo
}

// DecisionTreePredictor is a more lightweight version of DecisionTreeMetric
// It doesn't implement Move() and only worries about the prediction member of BestSplit()
// It is used to initialize a prediction for the root node, which may never get split.
type DecisionTreePredictor interface {
	Add(float64)
	Subtract(float64)
	Count() int
	Prediction() float64
	Metric() float64
	Copy() DecisionTreePredictor
}

type DecisionTreeMetricFactory interface {
	New(minLeafSize int) DecisionTreeMetric
	NewPredictionAccumulator() DecisionTreePredictor
}

// Types

// NumericSplitter is an implementation of Splitter that performs a simple split of an ordered feature variable.
type NumericSplitter float64

func (s NumericSplitter) Split(x float64) bool { return x < float64(s) }
func (s NumericSplitter) String() string       { return fmt.Sprintf("< %g", float64(s)) }

type Prediction struct {
	size       int
	prediction float64
}

// DecisionTreeNodetype describes an arbitrary node in a decision tree.
type DecisionTreeNode struct {
	Prediction

	// Left and Right children (nil values means this is a leaf)
	Left, Right *DecisionTreeNode

	// Feature index to split on.  The value -1 represents a leaf
	feature int

	// Reduction metric achieved by this split.
	reduction float64

	// Splitter to use on above feature if feature>= 0 else nil.
	splitter Splitter
}

func (n *DecisionTreeNode) Importances(importances []float64) {
	if n.feature >= 0 {
		importances[n.feature] += n.reduction
		n.Left.Importances(importances)
		n.Right.Importances(importances)
	}
}

// Dump prints a readable representation of a decision tree
func (n *DecisionTreeNode) Dump(w io.Writer, level int, prefix string) {
	for i := 0; i < level; i++ {
		fmt.Fprint(w, " ")
	}
	fmt.Fprintf(w, "%sprediction: %g feature: %d reduction: %g size: %d", prefix, n.prediction, n.feature, n.reduction, n.size)
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
	splitter  Splitter
	reduction float64

	left  Prediction
	right Prediction
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

type MSEMetricFactory struct {
	minLeafSize int
}

func NewMSEMetricFactory(minLeafSize int) MSEMetricFactory {
	return MSEMetricFactory{minLeafSize}
}

func (af MSEMetricFactory) New(minLeafSize int) DecisionTreeMetric {
	return NewMSEMetric(minLeafSize)
}

func (af MSEMetricFactory) NewPredictionAccumulator() DecisionTreePredictor {
	return NewMeanPredictor()
}

type MeanPredictor struct {
	varianceAccumulator *stats.VarianceAccumulator
}

func NewMeanPredictor() DecisionTreePredictor {
	return &MeanPredictor{stats.NewVarianceAccumulator()}
}

func (mp MeanPredictor) Add(x float64) {
	mp.varianceAccumulator.Add(x)
}

func (mp MeanPredictor) Subtract(x float64) {
	mp.varianceAccumulator.Subtract(x)
}

func (mp MeanPredictor) Count() int {
	return mp.varianceAccumulator.Count()
}

func (mp MeanPredictor) Prediction() float64 {
	return mp.varianceAccumulator.Mean()
}

func (mp MeanPredictor) Metric() float64 {
	return mp.varianceAccumulator.Variance()
}

func (mp MeanPredictor) Copy() DecisionTreePredictor {
	return MeanPredictor{mp.varianceAccumulator.Copy()}
}

type MSEMetric struct {
	minLeafSize          int
	previousFeatureValue float64
	left, right          DecisionTreePredictor

	count          int
	bestMetric     float64
	bestSplitValue float64

	bestLeft  Prediction
	bestRight Prediction

	initialMetric float64
}

func NewMSEMetric(minLeafSize int) *MSEMetric {
	return &MSEMetric{
		left:                 NewMeanPredictor(),
		right:                NewMeanPredictor(),
		bestMetric:           math.Inf(1),
		previousFeatureValue: math.Inf(-1),
		count:                0,
		minLeafSize:          minLeafSize,
	}
}

// Add target value to "right" tally
func (a *MSEMetric) Add(dtp DecisionTreePredictor) {
	a.right = dtp
	a.count = dtp.Count()
}

// Evaluate the metric assuming a break to the immediate left of
// featureValue (feature Value is in right set).  Then move the
// attribute value from the right set to left set.
func (a *MSEMetric) Move(featureValue, targetValue float64) {
	// If left count is zero, the is the initial move and
	// the current metric is the one to beat.
	if a.left.Count() == 0 {
		a.initialMetric = a.right.Metric()
		a.bestMetric = a.initialMetric
	}
	if featureValue != a.previousFeatureValue { // End of a run of identical values
		metric := float64(a.left.Count())*a.left.Metric() + float64(a.right.Count())*a.right.Metric()
		if metric < a.bestMetric && a.left.Count() >= a.minLeafSize && a.right.Count() >= a.minLeafSize {
			a.bestMetric = metric
			a.bestSplitValue = 0.5 * (featureValue + a.previousFeatureValue)

			if a.bestLeft.size = a.left.Count(); a.bestLeft.size > 0 {
				a.bestLeft.prediction = a.left.Prediction()
			}

			if a.bestRight.size = a.right.Count(); a.bestRight.size > 0 {
				a.bestRight.prediction = a.right.Prediction()
			}
		}
		a.previousFeatureValue = featureValue
	}

	a.right.Subtract(targetValue)
	a.left.Add(targetValue)
}

func (a *MSEMetric) BestSplit() *SplitInfo {
	var result *SplitInfo

	if a.right.Count() != 0 {
		panic("BestSplit() called prematurely (fewer Move() calls than Add() calls)")
	}

	if a.bestLeft.size != 0 && a.bestRight.size != 0 {
		log.Printf("initialMetric: %v; bestMetric: %v", a.initialMetric, a.bestMetric)
		result = &SplitInfo{
			splitter:  NumericSplitter(a.bestSplitValue),
			reduction: a.initialMetric - a.bestMetric,
			left:      a.bestLeft,
			right:     a.bestRight,
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
	f OrderedFeature,
	target Feature,
	nodeMembership []int,
	bag Bag,
	splittableNodes []*DecisionTreeNode,
	accums []DecisionTreePredictor,
	minLeafSize int) []*SplitInfo {

	accumulators := make([]DecisionTreeMetric, len(accums))
	for i := range accums {
		accumulators[i] = NewMSEMetric(minLeafSize)
		accumulators[i].Add(accums[i].Copy())
	}

	// Second pass - move points from right to left and evalute new metric
	for i := range nodeMembership {
		iOrdered := f.InOrder(i)
		if nm := nodeMembership[iOrdered]; nm >= 0 {
			for j := 0; j < bag.Count(iOrdered); j++ {
				accumulators[nm].Move(f.NumericValue(iOrdered), target.NumericValue(iOrdered))
			}
		}
	}

	// Collect results from accumulators
	result := make([]*SplitInfo, len(splittableNodes))
	for i := range result {
		if bestSplit := accumulators[i].BestSplit(); bestSplit != nil {
			result[i] = bestSplit
		} else {
			result[i] = nil
		}
	}

	return result
}

type decisionTreeGrower struct {
	MaxFeatures int
	MinLeafSize int
}

func dtInitialize(target Feature, bag Bag, af DecisionTreeMetricFactory) ([]*DecisionTreeNode, []int) {
	node := &DecisionTreeNode{feature: -1}

	splittableNodeMembership := make([]int, bag.Len())
	acc := af.NewPredictionAccumulator()

	for i := 0; i < bag.Len(); i++ {
		splittableNodeMembership[i] = 0 // Root node
		for j := 0; j < bag.Count(i); j++ {
			acc.Add(target.NumericValue(i))
		}
	}

	node.Prediction = Prediction{size: acc.Count(), prediction: acc.Prediction()}

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
			if nodeCandidateSplits[inode] != nil {
				improvingSplits = append(improvingSplits, nodeCandidateSplits[inode])
			}
		}

		// Consider a random subset of feature splits of size maxFeatures and pick the best of those
		var bestSplit *FeatureSplitInfo
		for i := 0; i < maxFeatures && i < len(improvingSplits); i++ {
			irand := i + rand.Intn(len(improvingSplits)-i)
			improvingSplits[i], improvingSplits[irand] = improvingSplits[irand], improvingSplits[i]
			if bestSplit == nil || improvingSplits[i].reduction > bestSplit.reduction {
				bestSplit = improvingSplits[i]
			}
		}

		var newPair *SplitPair = nil
		if bestSplit != nil {
			node.feature = bestSplit.feature
			node.splitter = bestSplit.splitter
			node.reduction = bestSplit.reduction

			node.Left = &DecisionTreeNode{feature: -1, Prediction: bestSplit.left}
			node.Right = &DecisionTreeNode{feature: -1, Prediction: bestSplit.right}

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

func dtInitialMetrics(target Feature, nodeMembership []int, bag Bag, splittableNodes []*DecisionTreeNode, af DecisionTreeMetricFactory) []DecisionTreePredictor {
	accumulators := make([]DecisionTreePredictor, len(splittableNodes))
	for i := range accumulators {
		accumulators[i] = af.NewPredictionAccumulator()
	}

	// First pass - accumulate stats with all points to right of split
	// Add points from right to left so they are removed in LIFO order
	for i := 0; i < len(nodeMembership); i++ {
		if nm := nodeMembership[i]; nm >= 0 {
			for j := 0; j < bag.Count(i); j++ {
				accumulators[nm].Add(target.NumericValue(i))
			}
		}
	}

	return accumulators
}

type SplitPair struct{ left, right int }

func (dtg *decisionTreeGrower) grow(features []OrderedFeature, target Feature, bag Bag, oobPrediction []stats.Accumulator, af DecisionTreeMetricFactory) *DecisionTreeNode {
	maxFeatures := dtg.MaxFeatures
	if maxFeatures > len(features) || maxFeatures <= 0 {
		maxFeatures = len(features)
	}

	initialSplittableNodes, splittableNodeMembership := dtInitialize(target, bag, af)
	root := initialSplittableNodes[0]

	// candidateSplitsByFeature is a fixed length slices that is
	// re-used during each iteration
	candidateSplitsByFeature := make([][]*FeatureSplitInfo, len(features))

	var nextSplittableNodes []*DecisionTreeNode
	for splittableNodes := initialSplittableNodes; len(splittableNodes) > 0; splittableNodes = nextSplittableNodes {
		accums := dtInitialMetrics(target, splittableNodeMembership, bag, splittableNodes, af)
		log.Printf("accums: %v", accums)

		log.Printf("*** New Iteration ***:  splittableNodeMembership: %v", splittableNodeMembership)

		// For each feature find all optimal splits for that feature for each splittable node
		for i, feature := range features {
			candidateSplitsByFeature[i] = make([]*FeatureSplitInfo, 0, len(splittableNodes))
			log.Printf("Best splits by node, feature %d:\n", i)
			for _, dtos := range dtOptimalSplit(feature, target, splittableNodeMembership, bag, splittableNodes, accums, dtg.MinLeafSize) {
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
					if oobPrediction != nil && bag.Count(i) == 0 {
						oobPrediction[i].Add(splittableNode.prediction)
					}
				}
			}
		}
	}
	return root
}

func (dtg *DecisionTreeNode) Predict(features []Feature) []float64 {
	var result []float64
	if len(features) > 0 {
		result = make([]float64, features[0].Len())
		for i := range result {
			node := dtg
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

type DecisionTree struct {
	nFeatures int
	root      *DecisionTreeNode
	grower    *decisionTreeGrower
}

func NewDecisionTree() *DecisionTree {
	return &DecisionTree{
		0,
		nil,
		&decisionTreeGrower{MaxFeatures: math.MaxInt32, MinLeafSize: 1},
	}
}

func (dtr *DecisionTree) Importances() []float64 {
	fmt.Printf("Importances(): nFeatures: %d", dtr.nFeatures)
	importances := make([]float64, dtr.nFeatures)
	dtr.root.Importances(importances)
	return importances
}

func (dtr *DecisionTree) Fit(features []OrderedFeature, target Feature, af DecisionTreeMetricFactory) {
	for _, f := range features {
		f.Prepare()
	}

	bag := FullBag(features[0].Len())
	log.Printf("bag: %v", bag)

	// Pass nil to oob predictions because they are not applicable
	// to a single decision tree
	dtr.root = dtr.grower.grow(features, target, bag, nil, af)

	dtr.nFeatures = len(features)

	log.Print("Tree Dump:")
	dtr.Dump(os.Stderr)
}

func (dtr *DecisionTree) Predict(features []Feature) []float64 {
	var result []float64
	if dtr.root != nil {
		result = dtr.root.Predict(features)
	}
	return result
}

// Dump prints a readable representation of a decision tree
func (t *DecisionTree) Dump(w io.Writer) {
	fmt.Fprintf(w, "Trained with %d features\n", t.nFeatures)
	t.root.Dump(w, 0, "Root ")
}
