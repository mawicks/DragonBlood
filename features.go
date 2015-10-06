package DragonBlood

import (
	"math"
	"strconv"
)

type Feature interface {
	Length() int
	Add(interface{})
	AddFromString(string)
	Get(int) interface{}
	Name() string
}

// NumericFeature implements Feature
type NumericFeature struct {
	name   string
	values []float64
}

func NewNumericFeature(name string) *NumericFeature {
	return &NumericFeature{name, make([]float64, 0)}
}

func (nf *NumericFeature) Name() string { return nf.name }

func (nf *NumericFeature) Add(any interface{}) {
	value := math.NaN()

	switch any := any.(type) {
	case float64:
		value = any
	case float32:
		value = float64(any)
	case string:
		value, _ = strconv.ParseFloat(any, 64)
	}
	nf.values = append(nf.values, value)
}

func (nf *NumericFeature) AddFromString(stringValue string) {
	value, error := strconv.ParseFloat(stringValue, 64)
	if error != nil {
		value = math.NaN()
	}
	nf.Add(value)
}

func (nf *NumericFeature) Get(index int) interface{} { return nf.values[index] }

func (nf *NumericFeature) Length() int { return len(nf.values) }

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

func (nf *CategoricalFeature) Add(any interface{}) {
	// For now, only accept strings:
	s := any.(string)

	// Called for its side effects, so the return values are ignored
	nf.AddFromString(s)
}

func (nf *CategoricalFeature) AddFromString(s string) {
	// Called for its side effects, so the return values are ignored
	m, _ := nf.stringTable.Map(s)
	nf.values = append(nf.values, m)
}

func (nf *CategoricalFeature) Get(index int) interface{} {
	return nf.stringTable.Unmap(nf.values[index])
}

func (nf *CategoricalFeature) Length() int { return len(nf.values) }

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
