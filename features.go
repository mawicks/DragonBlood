package DragonBlood

import (
	"math"
	"sort"
	"strconv"
)

type Feature interface {
	Len() int
	Add(...interface{})
	AddFromString(...string)
	Name() string

	// Because Go type assertions are inefficient, all
	// features types can present values as float64.
	// For categorical valvues, this would be an integer representing the value.
	// Float64 can represent all int32 values exactly.
	// This is invertable via Unmap() so that
	// Decode(NumericValue(i)) == Value(i)
	NumericValue(int) float64
	Decode(float64) interface{}

	Value(int) interface{}
}

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

// NumericFeature implements Feature
type NumericFeature struct {
	name                string
	values              []float64
	attributeIndexSlice attributeIndexSlice
}

func NewNumericFeature(name string) *NumericFeature {
	return &NumericFeature{name, make([]float64, 0), nil}
}

func (nf *NumericFeature) Name() string { return nf.name }

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
	nf.attributeIndexSlice = make(attributeIndexSlice, 0)
	for i, v := range nf.values {
		nf.attributeIndexSlice = append(nf.attributeIndexSlice, attributeIndex{v, i})
	}
	sort.Sort(nf.attributeIndexSlice)
}

func (nf *NumericFeature) InOrder(index int) int {
	return nf.attributeIndexSlice[index].index
}

// CategoricalFeature implements Feature
type CategoricalFeature struct {
	name        string
	stringTable StringTable
	values      []int
}

func NewCategoricalFeature(name string, st StringTable) *CategoricalFeature {
	return &CategoricalFeature{name, st, make([]int, 0)}
}

func (nf *CategoricalFeature) Name() string { return nf.name }

func (nf *CategoricalFeature) Add(anyValues ...interface{}) {
	for _, any := range anyValues {
		// For now, only accept strings:
		s := any.(string)

		// Called for its side effects, so the return values are ignored
		nf.AddFromString(s)
	}
}

func (nf *CategoricalFeature) AddFromString(strings ...string) {
	for _, s := range strings {
		// Called for its side effects, so the return values are ignored
		m, _ := nf.stringTable.Encode(s)
		nf.values = append(nf.values, m)
	}
}

func (nf *CategoricalFeature) NumericValue(index int) float64 {
	return float64(nf.values[index])
}

func (nf *CategoricalFeature) Decode(x float64) interface{} {
	return nf.stringTable.Decode(int(x))
}

func (nf *CategoricalFeature) Value(index int) interface{} {
	return nf.stringTable.Decode(nf.values[index])
}

func (nf *CategoricalFeature) Len() int { return len(nf.values) }

// FeatureFactory
type FeatureFactory interface {
	New(name string) Feature
}

// Type: NumericFeatureFactory
type NumericFeatureFactory struct{}

func NewNumericFeatureFactory() FeatureFactory { return NumericFeatureFactory{} }

func (NumericFeatureFactory) New(name string) Feature {
	return NewNumericFeature(name)
}

// Type: CategoricalFeatureFactory
type CategoricalFeatureFactory struct{}

func NewCategoricalFeatureFactory() FeatureFactory { return CategoricalFeatureFactory{} }

func (CategoricalFeatureFactory) New(name string) Feature {
	return NewCategoricalFeature(name, NewStringTable())
}
