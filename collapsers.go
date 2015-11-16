package DragonBlood

import "fmt"

// Numeric Features
type DTNumericFeature struct {
	*NumericFeature
}

func NewDTNumericFeature(nf *NumericFeature) *DTNumericFeature {
	return &DTNumericFeature{nf}
}

type NumericCollapser struct {
	target DTTarget
}

func (nf *DTNumericFeature) NewCollapser(target DTTarget) DTCollapser {
	return &NumericCollapser{}
}

func (nc *NumericCollapser) Add(item int, count int) {}

func (nc *NumericCollapser) BestSplit(minLeafSize int) (si *SplitInfo) {
	return si
}

// Categorical Features
type DTCategoricalFeature struct {
	*CategoricalFeature
}

func NewDTCategoricalFeature(nf *CategoricalFeature) *DTCategoricalFeature {
	return &DTCategoricalFeature{nf}

}

type CategoricalCollapser struct {
	feature      *CategoricalFeature
	target       DTTarget
	accumulators []DTSplittingCriterion
}

func (cf *DTCategoricalFeature) NewCollapser(target DTTarget) DTCollapser {
	accumulators := make([]DTSplittingCriterion, cf.Range())
	for i := range accumulators {
		accumulators[i] = target.NewSplittingCriterion()
	}
	return &CategoricalCollapser{cf.CategoricalFeature, target, accumulators}
}

func (cc *CategoricalCollapser) Add(item int, count int) {
	for i := 0; i < count; i++ {
		cc.accumulators[int(cc.feature.NumericValue(item))].Add(cc.target.NumericValue(item))
	}
}

func (cc *CategoricalCollapser) BestSplit(minLeafSize int) (si *SplitInfo) {
	fmt.Printf("entry: CategoricalCollapser.BestSplit()\n")
	right := cc.target.NewSplittingCriterion()

	for _, acc := range cc.accumulators {
		right.Combine(acc)
		fmt.Printf("\tright: Metric(), Prediction()  = %g, %g\n", right.Metric(), right.Prediction())
	}

	best := right.Metric()
	fmt.Printf("best: (initialright metric): %v\n", best)

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
			si.splitter = NumericSplitter(float64(i) + 0.5)
		}
	}
	return si
}
