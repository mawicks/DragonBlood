package stats

type Accumulator interface {
	Add(float64) float64
	Subtract(float64) float64
	Value() float64
	Reset()
	Copy() Accumulator
}
