package DragonBlood

import (
	"math"
	"sort"
)

// Dense Numeric Feature Splitter A dense numeric feature is one where
// there is likely to be no advantage in collecting counts of the
// number of times each value occurs.  The overhead associated with
// creating a map for the count of each value of the feature is
// eliminated.

type DTNumericFeature struct {
	*NumericFeature
}

func NewDTNumericFeature(nf *NumericFeature) *DTNumericFeature {
	return &DTNumericFeature{nf}
}

type NumericSplitter struct {
	target DTTarget
	order  []int
	count  []int
	accum  DTSplittingCriterion
}

func (nf *DTNumericFeature) NewSplitter(target DTTarget, estSize int) DTSplitter {
	return &NumericSplitter{
		target: target,
		order:  make([]int, 0, estSize),
		count:  make([]int, 0, estSize),
		accum:  target.NewSplittingCriterion(),
	}
}

func (nc *NumericSplitter) Add(item int, count int) {
	for i := 0; i < count; i++ {
		nc.order = append(nc.order, item)
		nc.accum.Add(nc.target.NumericValue(item))
	}
}

func (nc *NumericSplitter) Len() int { return len(nc.order) }
func (nc *NumericSplitter) Less(i, j int) bool {
	return nc.target.NumericValue(nc.order[i]) < nc.target.NumericValue(nc.order[j])
}
func (nc *NumericSplitter) Swap(i, j int) {
	nc.order[i], nc.order[j] = nc.order[j], nc.order[i]
	nc.count[i], nc.count[j] = nc.count[j], nc.order[i]
}

func (nc *NumericSplitter) BestSplit(minLeafSize int) (si *SplitInfo) {
	sort.Sort(nc)
	initial := nc.accum.Metric()

	left := nc.target.NewSplittingCriterion()
	previousValue := math.NaN()

	for _, item := range nc.order {
		currentValue := nc.target.NumericValue(item)

		left.Add(currentValue)
		nc.accum.Subtract(currentValue)

		reduction := initial - (left.Metric() + nc.accum.Metric())

		if previousValue != math.NaN() && currentValue != previousValue && (si == nil || reduction > si.reduction) {
			if si == nil {
				si = &SplitInfo{}
			}
			si.reduction = reduction
			si.splitter = NumericSplit((previousValue + currentValue) / 2.0)
		}

		previousValue = currentValue
	}
	return si
}

// Categorical Features
type DTCategoricalFeature struct {
	*CategoricalFeature
}

func NewDTCategoricalFeature(nf *CategoricalFeature) *DTCategoricalFeature {
	return &DTCategoricalFeature{nf}

}

type CategoricalSplitter struct {
	feature      *CategoricalFeature
	target       DTTarget
	accumulators []DTSplittingCriterion
}

func (cf *DTCategoricalFeature) NewSplitter(target DTTarget, estNodeSize int) DTSplitter {
	accumulators := make([]DTSplittingCriterion, cf.Range())
	for i := range accumulators {
		accumulators[i] = target.NewSplittingCriterion()
	}
	return &CategoricalSplitter{cf.CategoricalFeature, target, accumulators}
}

func (cc *CategoricalSplitter) Add(item int, count int) {
	for i := 0; i < count; i++ {
		cc.accumulators[int(cc.feature.NumericValue(item))].Add(cc.target.NumericValue(item))
	}
}

func (cc *CategoricalSplitter) BestSplit(minLeafSize int) (si *SplitInfo) {
	right := cc.target.NewSplittingCriterion()

	for _, acc := range cc.accumulators {
		right.Combine(acc)
	}

	best := right.Metric()

	left := cc.target.NewSplittingCriterion()
	for i, acc := range cc.accumulators {
		left.Combine(acc)
		right.Separate(acc)
		criterion := left.Metric() + right.Metric()
		if criterion < best {
			if si == nil {
				si = &SplitInfo{}
			}
			si.reduction = best - criterion
			si.splitter = NumericSplit(float64(i) + 0.5)
		}
	}
	return si
}
