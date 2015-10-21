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

	NumericValue(int) float64
	Decode(float64) interface{}

	Value(int) interface{}
}

// OrderedFeature is an interface for processing a list of feature
// values in a particular order.  The implementation should ensure that the
// following code will process the values in their intended order:
//    feature.Sort()
//    for i:=0; i<len(feature); i++ {
//       doSomething(feature.Value(InOrder(i)))
//    }
// The order is unspecified other than that identical
// values of Value() appear consecutively in the sequence.
// In other words:
//    if feature.Value(InOrder(i)) == feature.Value(InOrder(j))
//    then feature.Value(InOrder(k)) == feature.Value(InOrder(i)) for all i <= k <= j
// Calling Sort() should affect only the value of InOrder(i); it
// should have no effect on the value of NumericValue(i) and Value(i)
// for any i
type OrderedFeature interface {
	Feature
	Sort()
	InOrder(int) int
}

// attributeIndex
type attributeIndex struct {
	attribute float64
	index     int
}

// attributeIndexSlice
type attributeIndexSlice []attributeIndex

func (rfp attributeIndexSlice) Swap(i, j int) { rfp[i], rfp[j] = rfp[j], rfp[i] }
func (rfp attributeIndexSlice) Len() int      { return len(rfp) }
func (rfp attributeIndexSlice) Less(i, j int) bool {
	if math.IsNaN(rfp[j].attribute) {
		return true
	} else if math.IsNaN(rfp[i].attribute) {
		return false
	} else {
		return rfp[i].attribute < rfp[j].attribute
	}
}

// NumericFeature implements OrderedFeature (and Feature)
type NumericFeature struct {
	values     []float64
	orderIndex attributeIndexSlice
}

func NewNumericFeature(x []float64) *NumericFeature {
	if x == nil {
		x = make([]float64, 0)
	}
	return &NumericFeature{x, nil}
}

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

func (nf *NumericFeature) Len() int { return len(nf.values) }

func (nf *NumericFeature) Sort() {
	nf.orderIndex = make(attributeIndexSlice, 0)
	for i, v := range nf.values {
		nf.orderIndex = append(nf.orderIndex, attributeIndex{v, i})
	}
	sort.Sort(nf.orderIndex)
}

func (nf *NumericFeature) InOrder(index int) int {
	return nf.orderIndex[index].index
}

type intAttributeIndex struct {
	attribute int
	index     int
}

type intAttributeIndexSlice []intAttributeIndex

// CategoricalFeature implements Feature
type CategoricalFeature struct {
	stringTable StringTable
	values      []int
	orderIndex  intAttributeIndexSlice
}

func NewCategoricalFeature(st StringTable) *CategoricalFeature {
	return &CategoricalFeature{st, make([]int, 0), nil}
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
		// Called for its side effects, so the return values are ignored
		m, _ := cf.stringTable.Encode(s)
		cf.values = append(cf.values, m)
	}
}

func (cf *CategoricalFeature) NumericValue(index int) float64 {
	return float64(cf.values[index])
}

func (cf *CategoricalFeature) Decode(x float64) interface{} {
	return cf.stringTable.Decode(int(x))
}

func (cf *CategoricalFeature) Value(index int) interface{} {
	return cf.stringTable.Decode(cf.values[index])
}

func (cf *CategoricalFeature) Len() int { return len(cf.values) }

func (ais intAttributeIndexSlice) Swap(i, j int)      { ais[i], ais[j] = ais[j], ais[i] }
func (ais intAttributeIndexSlice) Len() int           { return len(ais) }
func (ais intAttributeIndexSlice) Less(i, j int) bool { return ais[i].attribute < ais[j].attribute }

func (cf *CategoricalFeature) Sort() {
	cf.orderIndex = make(intAttributeIndexSlice, 0)
	for i, v := range cf.values {
		cf.orderIndex = append(cf.orderIndex, intAttributeIndex{v, i})
	}
	sort.Sort(cf.orderIndex)
}

func (cf *CategoricalFeature) InOrder(index int) int {
	return cf.orderIndex[index].index
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
	return NewNumericFeature(nil)
}

// Deprecated
type CategoricalFeatureFactory struct{}

// Deprecated
func NewCategoricalFeatureFactory() FeatureFactory { return CategoricalFeatureFactory{} }

// Deprecated
func (CategoricalFeatureFactory) New() Feature {
	return NewCategoricalFeature(NewStringTable())
}
