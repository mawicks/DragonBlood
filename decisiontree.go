package DragonBlood

import (
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
)

type DTCollapser interface {
	Add(item int)
	BestSplit() *SplitInfo
}

type DTFeature interface {
	Feature
	NewCollapser(target Feature, af DecisionTreeSplittingCriterionFactory) DTCollapser
}

// Interfaces
type Splitter interface {
	Split(float64) bool
	// Return interpretable representation of Splitter.
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

func NewDecisionTreeSplitEvaluator(af DecisionTreeSplittingCriterionFactory, dtp DecisionTreeSplittingCriterion, minLeafSize int) *DecisionTreeSplitEvaluator {
	return &DecisionTreeSplitEvaluator{
		left:                 af.New(),
		right:                dtp,
		initialMetric:        dtp.Metric(),
		bestMetric:           dtp.Metric(),
		bestLeftSize:         0,
		previousFeatureValue: math.Inf(-1),
		count:                dtp.Count(),
		minLeafSize:          minLeafSize,
	}
}

// Evaluate the metric assuming a break to the immediate left of
// featureValue (feature Value is in right set).  Then move the
// attribute value from the right set to left set.
func (a *DecisionTreeSplitEvaluator) Move(featureValue, targetValue float64) {
	// FIXME
	//	log.Printf("Move(): featureValue: %v targetValue: %v  left metric: %g right metric: %g combined: %g\n",
	//	featureValue, targetValue, a.left.Metric(), a.right.Metric(), a.left.Metric()+a.right.Metric())
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
		//		log.Printf("bestMetric: %v", a.bestMetric)
		result = &SplitInfo{
			splitter:  NumericSplitter(a.bestSplitValue),
			reduction: a.initialMetric - a.bestMetric,
		}
	}
	return result
}

func dtBestSplit(node dtSplittableNode, features []OrderedFeature, target Feature, bag Bag, af DecisionTreeSplittingCriterionFactory, minLeafSize, maxFeatures int) (fs *FeatureSplit) {
	if maxFeatures > len(features) {
		maxFeatures = len(features)
	}

	// Select a random subset of features of size maxFeatures
	randomFeatures := make([]int, len(features))
	for i := range features {
		randomFeatures[i] = i
	}
	for i := 0; i < maxFeatures; i++ {
		j := rand.Intn(len(features) - i)
		randomFeatures[i], randomFeatures[i+j] = randomFeatures[i+j], randomFeatures[i]
	}
	randomFeatures = randomFeatures[0:maxFeatures]

	// Find the random feature with the best optimal split
	for _, i := range randomFeatures {
		os := dtOptimalSplit(node, features[i], target, bag, af, minLeafSize)
		if fs == nil || (os != nil && os.reduction < fs.reduction) {
			fs = &FeatureSplit{i, *os}
		}
	}
	return fs
}

// dtOptimalSplit computes an optimal split for node f using the
// passed feature and target.  bag maps each unit to the number of
// times that unit occurs in the current bag.  minSize is the minimum
// size for a leaf node.
func dtOptimalSplit(
	node dtSplittableNode,
	feature OrderedFeature,
	target Feature,
	bag Bag,
	af DecisionTreeSplittingCriterionFactory,
	minLeafSize int) (result *SplitInfo) {

	// Second pass - move points from right to left and evalute new metric
	for i, nm := range node.members {
		iOrdered := feature.InOrder(i)
		if nm := node.members[iOrdered]; nm >= 0 {
			for j := 0; j < bag.Count(iOrdered); j++ {
				accum.Move(f.NumericValue(iOrdered), target.NumericValue(iOrdered))
			}
		}
	}

	// Collect results from accumulators
	if bestSplit := accum.BestSplit(); bestSplit != nil {
		result = bestSplit
	} else {
		result = nil
	}

	return result
}

type decisionTreeGrower struct {
	MaxFeatures int
	MinLeafSize int
}

type dtSplittableNode struct {
	*DecisionTreeNode
	members []int
}

func dtInitialize(target Feature, bag Bag, af DecisionTreeSplittingCriterionFactory) dtSplittableNode {
	nodeMembership := make([]int, bag.Len())

	for i := 0; i < bag.Len(); i++ {
		nodeMembership[i] = i // Root node
	}

	return dtSplittableNode{NewDecisionTreeNode(), nodeMembership}
}

func dtAddNodeMetrics(node dtSplittableNode, target Feature, bag Bag, af DecisionTreeSplittingCriterionFactory) {
	accumulator := af.New()

	// First pass - accumulate stats with all points
	for i := 0; i < len(node.members); i++ {
		nm := node.members[i]
		for j := 0; j < bag.Count(nm); j++ {
			accumulator.Add(target.NumericValue(nm))
		}
	}

	node.Prediction = Prediction{accumulator.Count(), accumulator.Prediction()}
}

type SplitPair struct{ left, right int }

func (dtg *decisionTreeGrower) grow(features []OrderedFeature, target Feature, bag Bag, oobPrediction []DecisionTreeSplittingCriterion, af DecisionTreeSplittingCriterionFactory) *DecisionTreeNode {
	maxFeatures := dtg.MaxFeatures
	if maxFeatures > len(features) || maxFeatures <= 0 {
		maxFeatures = len(features)
	}

	splittableRoot := dtInitialize(target, bag, af)

	// candidateSplitsByFeature is a fixed length slice that is
	// re-used during each iteration
	candidateSplitsByFeature := make([][]*FeatureSplit, len(features))

	var nextSplittableNodes []dtSplittableNode
	for splittableNodes := []dtSplittableNode{splittableRoot}; len(splittableNodes) > 0; splittableNodes = nextSplittableNodes {
		nextSplittableNodes = make([]dtSplittableNode, 0, len(splittableNodes))
		for _, node := range splittableNodes {
			bestSplit := dtBestSplit(node, features, target, bag, af, dtg.MinLeafSize, maxFeatures)
			dtAddNodeMetrics(node, target, bag, af)
			if bestSplit != nil { // Was an improving split found?
				node.FeatureSplit = *bestSplit

				node.Left = NewDecisionTreeNode()
				node.Right = NewDecisionTreeNode()

				rightOffset := 0
				for i, member := range node.members {
					if node.splitter.Split(features[node.feature].NumericValue(member)) { // Left
						node.members[rightOffset], node.members[i] = node.members[i], node.members[rightOffset]
						rightOffset += 1
					}
				}
				nextSplittableNodes = append(nextSplittableNodes, dtSplittableNode{node.Left, node.members[0:rightOffset]})
				nextSplittableNodes = append(nextSplittableNodes, dtSplittableNode{node.Right, node.members[rightOffset:]})

			} else { // node is a now a leaf node
				for _, member := range node.members {
					if oobPrediction != nil && bag.Count(member) == 0 {
						oobPrediction[member].Add(node.prediction)
					}
				}
			}
		}
	}

	return splittableRoot.DecisionTreeNode
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

// Parameter setters:
func (dtr *DecisionTree) SetMinLeafSize(m int) *DecisionTree {
	dtr.grower.MinLeafSize = m
	return dtr
}

func (dtr *DecisionTree) SetMaxFeatures(m int) *DecisionTree {
	dtr.grower.MaxFeatures = m
	return dtr
}

func (dtr *DecisionTree) Importances() []float64 {
	fmt.Printf("xImportances(): nFeatures: %d", dtr.nFeatures)
	importances := make([]float64, dtr.nFeatures)
	dtr.root.Importances(importances)
	return importances
}

func (dtr *DecisionTree) Fit(features []OrderedFeature, target Feature, af DecisionTreeSplittingCriterionFactory) {
	for _, f := range features {
		f.Prepare()
	}

	bag := FullBag(features[0].Len())

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
