package DragonBlood

import (
	"fmt"
	"math"

	"github.com/mawicks/DragonBlood/stats"
)

type DecisionTreeSplittingCriterion interface {
	// Add() updates the internal state associated with adding x to a node.
	Add(x float64)

	// Subtract() updates the internal state associated with removing x to a node.
	// Add() was previously called with x.
	Subtract(float64)

	// Count() returns the number of items currently in the node
	Count() int

	// Prediction() returns the node prediction associated with the current set
	Prediction() float64

	// Metric() returns the current metric associated with the node.
	// It should return a value weighted so that two metrics
	// from a split can be added together.  That way, the sum of
	// the metrics across all of the nodes is the metric for the
	// entire tree.  For example, sum-of-squared-error is a valid
	// metric, but mean-squared-error is not.
	Metric() float64

	// Copy() returns a new DecisionTreePredictor of the same type
	// with identical internal state.
	Copy() DecisionTreeSplittingCriterion

	Dump(string)
}

type DecisionTreeSplittingCriterionFactory interface {
	New() DecisionTreeSplittingCriterion
}

type MSECriterionFactory struct{}

func NewMSECriterionFactory() MSECriterionFactory {
	return MSECriterionFactory{}
}

func (cf MSECriterionFactory) New() DecisionTreeSplittingCriterion {
	return NewMSECriterion()
}

type MSECriterion struct {
	varianceAccumulator *stats.VarianceAccumulator
}

func NewMSECriterion() *MSECriterion {
	return &MSECriterion{stats.NewVarianceAccumulator()}
}

func (mp MSECriterion) Add(x float64) {
	mp.varianceAccumulator.Add(x)
}

func (mp MSECriterion) Subtract(x float64) {
	mp.varianceAccumulator.Subtract(x)
}

func (mp MSECriterion) Count() int {
	return mp.varianceAccumulator.Count()
}

func (mp MSECriterion) Prediction() float64 {
	return mp.varianceAccumulator.Mean()
}

func (mp MSECriterion) Metric() float64 {
	return mp.varianceAccumulator.SumSquaredError()
}

func (mp MSECriterion) Copy() DecisionTreeSplittingCriterion {
	return MSECriterion{mp.varianceAccumulator.Copy()}
}

func (mp MSECriterion) Dump(string) {}

// Definitions associated with entropy splitting criterion

type EntropyCriterionFactory struct{}

func NewEntropyCriterionFactory() EntropyCriterionFactory {
	return EntropyCriterionFactory{}
}

func (cf EntropyCriterionFactory) New() DecisionTreeSplittingCriterion {
	return NewCategoricalCriterion(func(p float64) float64 { return -math.Log2(float64(p)) })
}

type GiniCriterionFactory struct{}

func NewGiniCriterionFactory() GiniCriterionFactory {
	return GiniCriterionFactory{}
}
func (cf GiniCriterionFactory) New() DecisionTreeSplittingCriterion {
	return NewCategoricalCriterion(func(p float64) float64 { return (1 - p) })
}

type CategoricalCriterion struct {
	count  int
	counts map[float64]int
	// The categorical node metric is assumed to have the form sum_i count_i function(p_i)
	// For gini: function(p) = (1 - p)
	// For entropy: function(p) = - log2(p)
	function func(float64) float64
}

func NewCategoricalCriterion(f func(float64) float64) *CategoricalCriterion {
	return &CategoricalCriterion{
		0,
		make(map[float64]int, 10),
		f,
	}
}

func (ec *CategoricalCriterion) Add(x float64) {
	ec.counts[x] += 1
	ec.count += 1
}

func (ec *CategoricalCriterion) Subtract(x float64) {
	if ec.count <= 0 {
		panic("Subtract() without corresponding Add()")
	}
	if ec.counts[x] <= 0 {
		panic(fmt.Sprintf("Subtract() without corresponding Add() for item %v", x))
	}
	ec.count -= 1
	ec.counts[x] -= 1
	if ec.counts[x] <= 0 {
		delete(ec.counts, x)
	}
}

func (ec *CategoricalCriterion) Count() int {
	return ec.count
}

func (ec *CategoricalCriterion) Prediction() (prediction float64) {
	prediction = math.NaN()
	maxCount := 0

	for x, c := range ec.counts {
		if c > maxCount {
			maxCount = c
			prediction = x
		}
	}

	return prediction
}

func (ec *CategoricalCriterion) Metric() float64 {
	m := 0.0
	if ec.count > 0 {
		for _, c := range ec.counts {
			if c > 0 {
				p := float64(c) / float64(ec.count)
				m += float64(c) * ec.function(p)
			}
		}
	}
	return m
}

func (ec *CategoricalCriterion) Copy() DecisionTreeSplittingCriterion {
	new := NewCategoricalCriterion(ec.function)
	for k, v := range ec.counts {
		new.counts[k] = v
	}
	new.count = ec.count
	return new
}

func (ec *CategoricalCriterion) Dump(s string) {
	fmt.Printf("%s %+v\n", s, ec.counts)
}
