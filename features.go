package DragonBlood

import (
	"math"
	"strconv"
)

type Feature interface {
	Length() int
	Add(interface{})
	Get(int) interface{}
	Name() string
}

type FeatureFactory interface {
	New(name string) Feature
}

// Type: NumericFeature
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

// Type: NumericFeatureFactory
type NumericFeatureFactory struct{}

func NewNumericFeatureFactory() FeatureFactory { return NumericFeatureFactory{} }

func (NumericFeatureFactory) New(name string) Feature {
	return NewNumericFeature(name)
}
