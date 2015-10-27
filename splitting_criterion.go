package DragonBlood

import "github.com/mawicks/DragonBlood/stats"

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
}

type DecisionTreeSplittingCriterionFactory interface {
	New() DecisionTreeSplittingCriterion
}
type MSECriterionFactory struct {
}

func NewMSEMetricFactory(minLeafSize int) MSECriterionFactory {
	return MSECriterionFactory{}
}

func (af MSECriterionFactory) New() DecisionTreeSplittingCriterion {
	return NewMSECriterion()
}

type MSECriterion struct {
	varianceAccumulator *stats.VarianceAccumulator
}

func NewMSECriterion() DecisionTreeSplittingCriterion {
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
