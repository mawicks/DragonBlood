package DragonBlood

import (
	"math"
	"sort"
	"strconv"
)

// Because Go type assertions are inefficient, all feature types
// present values as float64 via NumericValue().  This allows all
// features (numerical or categorical) to be split or partitioned by a
// comparison to a float64 split value

// For categorical features, NumericValue() returns the integer encoding
// of the original value embedded into a float64.  A float64 can
// represent all int32 values exactly, so there is no loss of
// information using this representation.

// For categorical features, the value returned by NumericValue() may
// be mapped back to the the original feature value using the
// CategoricalFeature's Decode() method.  These methods ensure
// that Decode(int(NumericValue(i))) == Value(i).

// NumericValue(i) and Value(i) return values in the order in which
// the feature values were added, where the order is represented by i.
type Feature interface {
	Len() int
	Add(...interface{})
	AddFromString(...string)

	NumericValue(i int) float64
	Value(i int) interface{}
}

// NumericFeature implements Feature
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
func (nf *NumericFeature) Value(index int) interface{}    { return nf.values[index] }

func (nf *NumericFeature) Sort() {
	sort.Sort(nf)
}

func (nf *NumericFeature) InOrder(index int) int {
	return nf.order[index]
}

// CategoricalFeature implements Feature
type CategoricalFeature struct {
	codec  Codec
	values []int
}

func NewCategoricalFeature(anyValues ...interface{}) *CategoricalFeature {
	st := NewCodec()
	new := &CategoricalFeature{st, nil}
	new.Add(anyValues...)
	return new
}

func (cf *CategoricalFeature) Categories() int {
	return cf.codec.Len()
}

func (cf *CategoricalFeature) Range() int {
	return cf.codec.Len()
}

func (cf *CategoricalFeature) Len() int { return len(cf.values) }

func (cf *CategoricalFeature) Add(anyValues ...interface{}) {
	for _, any := range anyValues {
		m, _ := cf.codec.Encode(any)
		cf.values = append(cf.values, m)
	}
}

func (cf *CategoricalFeature) AddFromString(strings ...string) {
	for _, s := range strings {
		cf.Add(s)
	}
}

func (cf *CategoricalFeature) NumericValue(index int) float64 {
	return float64(cf.values[index])
}

func (cf *CategoricalFeature) Encode(any interface{}) int {
	encoding, _ := cf.codec.Encode(any)
	return encoding
}

func (cf *CategoricalFeature) Decode(i int) interface{} {
	return cf.codec.Decode(i)
}

func (cf *CategoricalFeature) Value(index int) interface{} {
	return cf.codec.Decode(cf.values[index])
}

func (cf *CategoricalFeature) OrderEncoding() {
	r := cf.codec.Len()

	// Sort the original values
	sorted := make([]interface{}, r)
	for i := range sorted {
		sorted[i] = cf.Decode(i)
	}
	sort.Sort(SortAny(sorted))

	// Create new encoding by adding them in sorted order
	newCodec := NewCodec()
	for _, v := range sorted {
		newCodec.Encode(v)
	}

	// Precompute mapping from old values to new values
	reencoding := make([]int, r)
	for i := range reencoding {
		reencoding[i], _ = newCodec.Encode(cf.codec.Decode(i))
	}

	newValues := make([]int, len(cf.values))
	for i, v := range cf.values {
		newValues[i] = reencoding[v]
	}

	// Replace the original values and the original codec
	cf.values = newValues
	cf.codec = newCodec
}

type SortAny []interface{}

func (any SortAny) Less(i, j int) bool {
	switch any[i].(type) {
	case int:
		return any[i].(int) < any[j].(int)
	case float64:
		return any[i].(float64) < any[j].(float64)
	case string:
		return any[i].(string) < any[j].(string)
	}
	return false
}
func (any SortAny) Len() int      { return len(any) }
func (any SortAny) Swap(i, j int) { any[i], any[j] = any[j], any[i] }
