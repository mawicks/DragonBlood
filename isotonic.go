package DragonBlood

import "fmt"

func Isotonic(attribute, target []float64) (x, y []float64) {
	if len(attribute) != len(target) {
		panic(fmt.Sprintf("Argument mismatch: len(x)=%d, but len(target)=%d", len(x), len(target)))
	}

	sortable := NewNumericFeature(attribute...)
	sortable.Prepare()

	type segment struct {
		start, value float64
		count        int
	}

	segs := make([]segment, 0, len(attribute))

	for i := range target {
		iOrdered := sortable.InOrder(i)
		segs = append(segs, segment{attribute[iOrdered], target[iOrdered], 1})
		for j := len(segs) - 1; j > 0 && segs[j].value <= segs[j-1].value; j-- {
			newCount := segs[j].count + segs[j-1].count
			segs[j-1].value = (segs[j].value*float64(segs[j].count) + segs[j-1].value*float64(segs[j-1].count)) / float64(newCount)
			segs[j-1].count = newCount
			segs = segs[:len(segs)-1]
		}
	}

	x = make([]float64, len(segs))
	y = make([]float64, len(segs))

	for i, s := range segs {
		x[i] = s.start
		y[i] = s.value
	}
	return x, y
}
