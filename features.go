package DragonBlood

import (
	"math"
	"strconv"
)

type NumericFeature struct {
	name   string
	values []float64
}

func NewNumericFeature(name string) *NumericFeature {
	return &NumericFeature{name, make([]float64, 0)}
}

func (nf *NumericFeature) Name() string { return nf.name }

func (nf *NumericFeature) Add(value float64) {
	nf.values = append(nf.values, value)
}

func (nf *NumericFeature) AddFromString(stringValue string) {
	value, error := strconv.ParseFloat(stringValue, 64)
	if error != nil {
		value = math.NaN()
	}
	nf.Add(value)
}

func (nf *NumericFeature) Get(index int) float64 { return nf.values[index] }

func (nf *NumericFeature) Length() int { return len(nf.values) }
