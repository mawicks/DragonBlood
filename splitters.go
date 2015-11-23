package DragonBlood

// Numeric Features
type DTNumericFeature struct {
	*NumericFeature
}

func NewDTNumericFeature(nf *NumericFeature) *DTNumericFeature {
	return &DTNumericFeature{nf}
}

type NumericSplitter struct {
	target DTTarget
}

func (nf *DTNumericFeature) NewSplitter(target DTTarget) DTSplitter {
	return &NumericSplitter{}
}

func (nc *NumericSplitter) Add(item int, count int) {}

func (nc *NumericSplitter) BestSplit(minLeafSize int) (si *SplitInfo) {
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

func (cf *DTCategoricalFeature) NewSplitter(target DTTarget) DTSplitter {
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
