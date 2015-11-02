package DragonBlood

import (
	"math"
	"sort"
	"strconv"
)

// Because Go type assertions are inefficient, all
// features types can present values as float64 via NumericValue().
// For categorical values, this would be an integer (represented as a float64) corresponding to the value.
// A float64 can represent all int32 values exactly.
// The value returned by NumericValue() is invertable via Decode() so that
// Decode(NumericValue(i)) == Value(i).
// NumericValue(i) and Value(i) must retrieve values
// in the order in which they were added, where the order is represented by i.
type Feature interface {
	Len() int
	Add(...interface{})
	AddFromString(...string)

	NumericValue(i int) float64
	Decode(float64) interface{}

	Value(i int) interface{}
}

// OrderedFeature is an interface for processing a list of feature
// values in a particular order.  The implementation should ensure that the
// following code will process the values in their intended order:
//    feature.Prepare()
//    for i:=0; i<len(feature); i++ {
//       doSomething(feature.Value(InOrder(i)))
//    }
// The order is unspecified other than that identical
// values of Value() appear consecutively in the sequence.
// In other words:
//    if feature.Value(InOrder(i)) == feature.Value(InOrder(j))
//    then feature.Value(InOrder(k)) == feature.Value(InOrder(i)) for all i <= k <= j
// Calling Prepare() should affect only the value of InOrder(i); it
// should have no effect on the value of NumericValue(i) and Value(i)
// for any i
type OrderedFeature interface {
	Feature
	Prepare()
	InOrder(int) int
}

// NumericFeature implements OrderedFeature (and Feature)
type NumericFeature struct {
	values []float64
	order  []int
}

func NewNumericFeature(x ...float64) *NumericFeature {
	order := make([]int, 0, len(x))
	values := make([]float64, 0, len(x))
	for _, x := range x {
		order = append(order, len(order))
		values = append(values, x)
	}
	return &NumericFeature{values, order}
}

func (nf *NumericFeature) Swap(i, j int) { nf.order[i], nf.order[j] = nf.order[j], nf.order[i] }

func (nf *NumericFeature) Less(i, j int) bool {
	if math.IsNaN(nf.values[nf.order[j]]) {
		return true
	} else if math.IsNaN(nf.values[nf.order[i]]) {
		return false
	} else {
		return nf.values[nf.order[i]] < nf.values[nf.order[j]]
	}
}

func (nf *NumericFeature) Len() int { return len(nf.values) }

func (nf *NumericFeature) Add(anyValues ...interface{}) {
	value := math.NaN()

	for _, any := range anyValues {
		switch any := any.(type) {
		case float64:
			value = any
		case float32:
			value = float64(any)
		case int:
			value = float64(any)
		case string:
			value, _ = strconv.ParseFloat(any, 64)
		}
		nf.values = append(nf.values, value)
		nf.order = append(nf.order, len(nf.order))
	}
}

func (nf *NumericFeature) AddFromString(stringValues ...string) {
	for _, sv := range stringValues {
		value, error := strconv.ParseFloat(sv, 64)
		if error != nil {
			value = math.NaN()
		}
		nf.Add(value)
	}
}

func (nf *NumericFeature) NumericValue(index int) float64 { return nf.values[index] }
func (nf *NumericFeature) Decode(x float64) interface{}   { return x }
func (nf *NumericFeature) Value(index int) interface{}    { return nf.values[index] }

func (nf *NumericFeature) Prepare() {
	sort.Sort(nf)
}

func (nf *NumericFeature) InOrder(index int) int {
	return nf.order[index]
}

// CategoricalFeature implements Feature
type CategoricalFeature struct {
	codec  Codec
	values []int
	order  []int
}

func NewCategoricalFeature(anyValues ...interface{}) *CategoricalFeature {
	st := NewCodec()
	new := &CategoricalFeature{st, nil, nil}
	new.Add(anyValues...)
	return new
}

func (cf *CategoricalFeature) Swap(i, j int) { cf.order[i], cf.order[j] = cf.order[j], cf.order[i] }
func (cf *CategoricalFeature) Len() int      { return len(cf.values) }
func (cf *CategoricalFeature) Less(i, j int) bool {
	return cf.values[cf.order[i]] < cf.values[cf.order[j]]
}

func (cf *CategoricalFeature) Categories() int {
	return cf.codec.Len()
}

func (cf *CategoricalFeature) Add(anyValues ...interface{}) {
	for _, any := range anyValues {
		// For now, only accept strings:
		s := any.(string)

		// Called for its side effects, so the return values are ignored
		cf.AddFromString(s)
	}
}

func (cf *CategoricalFeature) AddFromString(strings ...string) {
	for _, s := range strings {
		m, _ := cf.codec.Encode(s)
		cf.values = append(cf.values, m)
		cf.order = append(cf.order, len(cf.order))
	}
}

func (cf *CategoricalFeature) NumericValue(index int) float64 {
	return float64(cf.values[index])
}

func (cf *CategoricalFeature) Decode(x float64) interface{} {
	return cf.codec.Decode(int(x))
}

func (cf *CategoricalFeature) Value(index int) interface{} {
	return cf.codec.Decode(cf.values[index])
}

func (cf *CategoricalFeature) Prepare() {
	sort.Sort(cf)
}

func (cf *CategoricalFeature) InOrder(index int) int {
	return cf.order[index]
}

func (cf *CategoricalFeature) Range() int {
	return cf.codec.Len()
}

// Deprecated
type FeatureFactory interface {
	New() Feature
}

// Deprecated
type NumericFeatureFactory struct{}

// func NewNumericFeatureFactory() FeatureFactory { return NumericFeatureFactory{} }

// Deprecated
func (NumericFeatureFactory) New(name string) Feature {
	return NewNumericFeature()
}

// Deprecated
type CategoricalFeatureFactory struct{}

// Deprecated
func NewCategoricalFeatureFactory() FeatureFactory { return CategoricalFeatureFactory{} }

// Deprecated
func (CategoricalFeatureFactory) New() Feature {
	return NewCategoricalFeature(NewCodec())
}
