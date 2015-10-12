package DragonBlood

import (
	"fmt"
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
	feature  int
	splitter Splitter
	metric   float64

	leftSize, rightSize             int
	leftPrediction, rightPrediction float64
	leftMetric, rightMetric         float64
}

type MSEAccumulator struct {
	left, right stats.SSEAccumulator
	count       int
}

// Add to "right" tally
func (a *MSEAccumulator) Add(t float64, count int) (mse, prediction float64) {
	for i := 0; i < count; i++ {
		a.right.Add(t)
	}
	a.count += count

	return a.right.MSE(), a.right.Mean()
}

// Move an  from right to left. Return resulting composite metric for this split.
func (a *MSEAccumulator) Move(t float64, count int) float64 {
	for i := 0; i < count; i++ {
		a.right.Subtract(t)
		a.left.Add(t)
	}
	return (a.left.Value() + a.right.Value()) / float64(a.count)
}

func optimalSplit(f DecisionTreeFeature,
	target DecisionTreeTarget,
	splittableNodes []*DecisionTreeNode,
	nodeMembership []int,
	bag Bag,
	minSize int) []*SplitInfo {
	result := make([]*SplitInfo, len(splittableNodes))
	return result
}

func isSplittable(node *DecisionTreeNode, minSize int) bool {
	return node.size > 2*minSize
}

type DecisionTree struct{}

func (dt *DecisionTree) Fit(features []DecisionTreeFeature, target DecisionTreeTarget, bag Bag, maxFeatures int, minSize int) *DecisionTreeNode {
	node := &DecisionTreeNode{}

	if maxFeatures > len(features) {
		maxFeatures = len(features)
	}

	rootNodeSplittable := isSplittable(node, minSize)

	// nextSplittableNodes is next generation of splittableNodes.
	// It is initialized here (and re-generated during each
	// iteration) and contains pointers to nodes that are eligible
	// for splitting during next generation.
	initialSplittableNodes := make([]*DecisionTreeNode, 0, 1)
	if rootNodeSplittable {
		initialSplittableNodes = append(initialSplittableNodes, node)
	}

	splittableNodeMembership := make([]int, len(bag))
	acc := stats.NewSSEAccumulator()
	for i, b := range bag {
		if b > 0 && rootNodeSplittable {
			splittableNodeMembership[i] = 0 // Root node
		} else {
			splittableNodeMembership[i] = -1 // An impossible node reference
		}
		if b > 0 {
			for j := 0; j < b; j++ {
				acc.Add(target.NumericValue(i))
			}
		}
	}

	node.prediction = acc.Mean()
	node.metric = acc.MSE()

	type SplitPair struct{ left, right int }

	// candidateSplittsByFeature is a fixed length slices that is
	// re-used during each iteration
	candidateSplitsByFeature := make([][]*SplitInfo, len(features))

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
			candidateSplitsByFeature[i] = optimalSplit(feature, target, splittableNodes, splittableNodeMembership, bag, minSize)
		}
		candidateSplits := make([]*SplitInfo, 0, len(features))
		bestSplits := make([]*SplitInfo, 0, len(splittableNodes))

		for inode, node := range splittableNodes {
			candidateSplits = candidateSplits[:0]
			for _, nodeCandidateSplits := range candidateSplitsByFeature {
				if nodeCandidateSplits[inode].metric < node.metric {
					candidateSplits = append(candidateSplits, nodeCandidateSplits[inode])
				}
			}

			var bestSplit *SplitInfo
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

				leftIndex := -1
				if isSplittable(leftChild, minSize) {
					leftIndex = len(nextSplittableNodes)
					nextSplittableNodes = append(nextSplittableNodes, leftChild)
				}

				rightIndex := -1
				if isSplittable(rightChild, minSize) {
					rightIndex = len(nextSplittableNodes)
					nextSplittableNodes = append(nextSplittableNodes, rightChild)
				}

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
				} else { // No split exists even though node passed isSplittable() test
					splittableNodeMembership[i] = -1 // An impossible node reference
				}
			}
		}
	}
	return node
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
