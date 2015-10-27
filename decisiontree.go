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

// Types

// NumericSplitter is an implementation of Splitter that performs a
// simple split of an ordered feature variable.
type NumericSplitter float64

func (s NumericSplitter) Split(x float64) bool { return x < float64(s) }
func (s NumericSplitter) String() string       { return fmt.Sprintf("< %g", float64(s)) }

type Prediction struct {
	size       int
	prediction float64
}

type SplitInfo struct {
	splitter  Splitter
	reduction float64
}

type FeatureSplit struct {
	feature int
	SplitInfo
}

// DecisionTreeNodetype describes any node in a decision tree.
type DecisionTreeNode struct {
	Prediction

	FeatureSplit

	// Left and Right children (nil values means this is a leaf)
	Left, Right *DecisionTreeNode
}

func NewDecisionTreeNode() *DecisionTreeNode {
	node := DecisionTreeNode{}
	node.feature = -1
	return &node
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
	fmt.Fprintf(w, "%sprediction: %g feature: %d reduction: %g size: %d",
		prefix, n.prediction, n.feature, n.reduction, n.size)
	if n.feature < 0 {
		fmt.Fprint(w, " (LEAF)")
	}
	fmt.Fprint(w, "\n")
	if n.Left != nil {
		n.Left.Dump(w, level+4, fmt.Sprintf("L feature_%d %s ",
			n.feature, n.splitter))
	}
	if n.Right != nil {
		n.Right.Dump(w, level+4, fmt.Sprintf("R feature_%d not %s ",
			n.feature, n.splitter))
	}
}

func (fsi *FeatureSplit) Dump() {
	if fsi != nil {
		log.Printf("\tfeature %v: %+v\n", fsi.feature, fsi.SplitInfo)
	} else {
		log.Printf("\tnil\n")
	}
}

type DecisionTreeSplitEvaluator struct {
	minLeafSize          int
	previousFeatureValue float64

	left, right DecisionTreeSplittingCriterion

	count         int
	initialMetric float64

	bestMetric     float64
	bestSplitValue float64
	bestLeftSize   int
}

func NewDecisionTreeSplitEvaluator(dtp DecisionTreeSplittingCriterion, minLeafSize int) *DecisionTreeSplitEvaluator {
	return &DecisionTreeSplitEvaluator{
		left:                 NewMSECriterion(),
		right:                dtp,
		bestMetric:           math.Inf(1),
		previousFeatureValue: math.Inf(-1),
		count:                dtp.Count(),
		minLeafSize:          minLeafSize,
	}
}

// Evaluate the metric assuming a break to the immediate left of
// featureValue (feature Value is in right set).  Then move the
// attribute value from the right set to left set.
func (a *DecisionTreeSplitEvaluator) Move(featureValue, targetValue float64) {
	// If left count is zero, the is the initial move and
	// the current metric is the one to beat.
	if a.left.Count() == 0 {
		a.initialMetric = a.right.Metric()
		a.bestMetric = a.initialMetric
		a.bestLeftSize = 0
	}
	if featureValue != a.previousFeatureValue { // End of a run of identical values
		metric := a.left.Metric() + a.right.Metric()
		if metric < a.bestMetric && a.left.Count() >= a.minLeafSize && a.right.Count() >= a.minLeafSize {
			a.bestMetric = metric
			a.bestSplitValue = 0.5 * (featureValue + a.previousFeatureValue)
			a.bestLeftSize = a.left.Count()
		}
		a.previousFeatureValue = featureValue
	}

	a.right.Subtract(targetValue)
	a.left.Add(targetValue)
}

func (a *DecisionTreeSplitEvaluator) BestSplit() *SplitInfo {
	var result *SplitInfo

	if a.right.Count() != 0 {
		panic("BestSplit() called prematurely (fewer Move() calls than Add() calls)")
	}

	if a.bestLeftSize > 0 && a.bestLeftSize < a.count {
		log.Printf("initialMetric: %v; bestMetric: %v", a.initialMetric, a.bestMetric)
		result = &SplitInfo{
			splitter:  NumericSplitter(a.bestSplitValue),
			reduction: a.initialMetric - a.bestMetric,
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
	accums []DecisionTreeSplittingCriterion, minLeafSize int) []*SplitInfo {

	accumulators := make([]*DecisionTreeSplitEvaluator, len(accums))
	for i := range accums {
		accumulators[i] = NewDecisionTreeSplitEvaluator(accums[i].Copy(), minLeafSize)
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

func dtInitialize(target Feature, bag Bag, af DecisionTreeSplittingCriterionFactory) (root *DecisionTreeNode, nodeMembership []int) {
	nodeMembership = make([]int, bag.Len())

	for i := 0; i < bag.Len(); i++ {
		nodeMembership[i] = 0 // Root node
		for j := 0; j < bag.Count(i); j++ {
		}
	}

	return NewDecisionTreeNode(), nodeMembership
}

func dtInitialMetrics(target Feature, nodeMembership []int, bag Bag, splittableNodes []*DecisionTreeNode, af DecisionTreeSplittingCriterionFactory) []DecisionTreeSplittingCriterion {
	accumulators := make([]DecisionTreeSplittingCriterion, len(splittableNodes))
	for i := range accumulators {
		accumulators[i] = af.New()
	}

	// First pass - accumulate stats with all points
	for i := 0; i < len(nodeMembership); i++ {
		if nm := nodeMembership[i]; nm >= 0 {
			for j := 0; j < bag.Count(i); j++ {
				accumulators[nm].Add(target.NumericValue(i))
			}
		}
	}

	for i, acc := range accumulators {
		splittableNodes[i].Prediction = Prediction{acc.Count(), acc.Prediction()}
	}

	for _, n := range splittableNodes {
		fmt.Printf("node: %+v\n", n)
	}
	return accumulators
}

type SplitPair struct{ left, right int }

func dtSelectSplits(
	splittableNodes []*DecisionTreeNode,
	candidateSplitsByFeature [][]*FeatureSplit,
	maxFeatures int) ([]*DecisionTreeNode, []*SplitPair) {

	nextSplittableNodes := make([]*DecisionTreeNode, 0, 2*len(splittableNodes))
	// nodeSplits is generated during each iteration and
	// contains the next-generation indexes of children of
	// nodes to be split (left and right values of
	// SplitPair are indexes of nextSplittableNodes; index
	// of nodeSplits match those of splittableNodes))
	nodeSplits := make([]*SplitPair, 0, len(splittableNodes))

	improvingSplits := make([]*FeatureSplit, 0, len(candidateSplitsByFeature))
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
		var bestSplit *FeatureSplit
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

			node.Left = NewDecisionTreeNode()
			node.Right = NewDecisionTreeNode()

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

func (dtg *decisionTreeGrower) grow(features []OrderedFeature, target Feature, bag Bag, oobPrediction []stats.Accumulator, af DecisionTreeSplittingCriterionFactory) *DecisionTreeNode {
	maxFeatures := dtg.MaxFeatures
	if maxFeatures > len(features) || maxFeatures <= 0 {
		maxFeatures = len(features)
	}

	root, splittableNodeMembership := dtInitialize(target, bag, af)

	// candidateSplitsByFeature is a fixed length slices that is
	// re-used during each iteration
	candidateSplitsByFeature := make([][]*FeatureSplit, len(features))

	var nextSplittableNodes []*DecisionTreeNode
	for splittableNodes := []*DecisionTreeNode{root}; len(splittableNodes) > 0; splittableNodes = nextSplittableNodes {
		log.Printf("*** New Iteration ***:  splittableNodeMembership: %v", splittableNodeMembership)

		accums := dtInitialMetrics(target, splittableNodeMembership, bag, splittableNodes, af)
		log.Printf("accums: %v", accums)

		// For each feature find all optimal splits for that feature for each splittable node
		for i, feature := range features {
			candidateSplitsByFeature[i] = make([]*FeatureSplit, 0, len(splittableNodes))
			log.Printf("Best splits by node, feature %d:\n", i)
			for _, dtos := range dtOptimalSplit(feature, target, splittableNodeMembership, bag, splittableNodes, accums, dtg.MinLeafSize) {
				var split *FeatureSplit
				if dtos != nil {
					split = &FeatureSplit{i, *dtos}
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

func (dtr *DecisionTree) Fit(features []OrderedFeature, target Feature, af DecisionTreeSplittingCriterionFactory) {
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
