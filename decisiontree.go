package DragonBlood

import (
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
)

type DTFeature interface {
	Feature
	NewSplitter(target DTTarget) DTSplitter
}

type DTSplitter interface {
	Add(item int, count int)
	BestSplit(minLeafSize int) *SplitInfo
}

type DTTarget interface {
	Feature
	NewSplittingCriterion() DTSplittingCriterion
}

// Interfaces
type Split interface {
	Split(float64) bool
	// Return interpretable representation of Split
	String() string
}

// Types

// NumericSplit is an implementation of Split that performs a
// simple split of an ordered feature variable.
type NumericSplit float64

func (s NumericSplit) Split(x float64) bool { return x < float64(s) }
func (s NumericSplit) String() string       { return fmt.Sprintf("< %g", float64(s)) }

type Prediction struct {
	size       int
	prediction float64
}

type SplitInfo struct {
	splitter  Split
	reduction float64
}

type FeatureSplit struct {
	feature int
	SplitInfo
}

// DTNodetype describes any node in a decision tree.
type DTNode struct {
	Prediction

	FeatureSplit

	// Left and Right children (nil values means this is a leaf)
	Left, Right *DTNode
}

func NewDTNode() *DTNode {
	node := DTNode{}
	node.feature = -1
	return &node
}

func (n *DTNode) Importances(importances []float64) {
	if n.feature >= 0 {
		importances[n.feature] += n.reduction
		n.Left.Importances(importances)
		n.Right.Importances(importances)
	}
}

// Dump prints a readable representation of a decision tree
func (n *DTNode) Dump(w io.Writer, level int, prefix string) {
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

type DTSplitEvaluator struct {
	minLeafSize          int
	previousFeatureValue float64

	left, right DTSplittingCriterion

	count         int
	initialMetric float64

	bestMetric     float64
	bestSplitValue float64
	bestLeftSize   int
}

func NewDTSplitEvaluator(af DTSplittingCriterionFactory, dtp DTSplittingCriterion, minLeafSize int) *DTSplitEvaluator {
	return &DTSplitEvaluator{
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
func (a *DTSplitEvaluator) Move(featureValue, targetValue float64) {
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

func (a *DTSplitEvaluator) BestSplit() *SplitInfo {
	var result *SplitInfo

	if a.right.Count() != 0 {
		panic("BestSplit() called prematurely (fewer Move() calls than Add() calls)")
	}

	if a.bestLeftSize > 0 && a.bestLeftSize < a.count {
		//		log.Printf("bestMetric: %v", a.bestMetric)
		if result == nil {
			result = &SplitInfo{}
		}
		result.splitter = NumericSplit(a.bestSplitValue)
		result.reduction = a.initialMetric - a.bestMetric
	}
	return result
}

// dtBestSplit finds the best improving split (if any) for this node over a random
// selection of maxFeatures items from the features slice
func dtBestSplit(node dtSplittableNode, features []DTFeature, target DTTarget, bag Bag, minLeafSize, maxFeatures int) (fs *FeatureSplit) {
	if len(node.members) <= minLeafSize {
		return fs
	}

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
	fmt.Printf("randomFeatures: %+v\n", randomFeatures)

	// Find the random feature with the best optimal split
	for i := range randomFeatures {
		collapser := features[i].NewSplitter(target)

		for _, nm := range node.members {
			if count := bag.Count(nm); count > 0 {
				collapser.Add(nm, count)
			}
		}

		if bs := collapser.BestSplit(minLeafSize); bs != nil && (fs == nil || bs.reduction < fs.reduction) {
			fs = &FeatureSplit{i, *bs}
		}

	}
	return fs
}

type decisionTreeGrower struct {
	MaxFeatures int
	MinLeafSize int
}

type dtSplittableNode struct {
	*DTNode
	members []int
}

func dtInitialize(target Feature, bag Bag) dtSplittableNode {
	nodeMembership := make([]int, bag.Len())

	for i := 0; i < bag.Len(); i++ {
		nodeMembership[i] = i // Root node
	}

	return dtSplittableNode{NewDTNode(), nodeMembership}
}

func dtAddNodeMetrics(node dtSplittableNode, target DTTarget, bag Bag) {
	accumulator := target.NewSplittingCriterion()

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

func (dtg *decisionTreeGrower) grow(features []DTFeature, target DTTarget, bag Bag, oobPrediction []DTSplittingCriterion) *DTNode {
	maxFeatures := dtg.MaxFeatures
	if maxFeatures > len(features) || maxFeatures <= 0 {
		maxFeatures = len(features)
	}

	splittableRoot := dtInitialize(target, bag)
	fmt.Printf("splittableRoot: %+v\n", splittableRoot)

	var nextSplittableNodes []dtSplittableNode
	for splittableNodes := []dtSplittableNode{splittableRoot}; len(splittableNodes) > 0; splittableNodes = nextSplittableNodes {
		fmt.Printf("splittableNodes: %+v\n\n", splittableNodes)
		nextSplittableNodes = make([]dtSplittableNode, 0, len(splittableNodes))
		for _, node := range splittableNodes {
			fmt.Printf("node: %+v\n", node)
			bestSplit := dtBestSplit(node, features, target, bag, dtg.MinLeafSize, maxFeatures)
			fmt.Printf("bestSplit: %+v\n\n", bestSplit)
			dtAddNodeMetrics(node, target, bag)

			if bestSplit != nil { // Was an improving split found?
				node.FeatureSplit = *bestSplit

				node.Left = NewDTNode()
				node.Right = NewDTNode()

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

	return splittableRoot.DTNode
}

func (dtg *DTNode) Predict(features []Feature) []float64 {
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
	root      *DTNode
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

func (dtr *DecisionTree) Fit(features []DTFeature, target DTTarget) {
	bag := FullBag(features[0].Len())

	// Pass nil to oob predictions because they are not applicable
	// to a single decision tree
	dtr.root = dtr.grower.grow(features, target, bag, nil)

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
