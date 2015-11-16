package DragonBlood

import "fmt"

// RocArea computes the area under the ROC curve
// score is any score that orders the data (e.g. a probability estimate)
// target contains the labels (True == positive; False == negative)
func ROCArea(score []float64, target []bool) float64 {
	if len(score) != len(target) {
		panic(fmt.Sprintf("Argument mismatch: len(score)=%d, but len(target)=%d", len(score), len(target)))
	}

	area := uint64(0)
	sortableScore := NewNumericFeature(score...)
	sortableScore.Sort()

	nPositive := 0
	nNegative := 0

	for i := len(target) - 1; i >= 0; i-- {
		if target[sortableScore.InOrder(i)] {
			nPositive += 1
		} else {
			nNegative += 1
			area += uint64(nPositive)
		}
	}
	return float64(area) / float64(nPositive) / float64(nNegative)
}

func MSE(score, target []float64) float64 {
	if len(score) != len(target) {
		panic(fmt.Sprintf("Argument mismatch: len(score)=%d, but len(target)=%d", len(score), len(target)))
	}

	accumulator := 0.0

	for i, t := range target {
		e := t - score[i]
		accumulator += e * e
	}

	return accumulator / float64(len(score))
}

func Accuracy(score, target []float64) float64 {
	if len(score) != len(target) {
		panic(fmt.Sprintf("Argument mismatch: len(score)=%d, but len(target)=%d", len(score), len(target)))
	}

	success := 0

	for i, t := range target {
		if score[i] == t {
			success += 1
		}
	}

	return float64(success) / float64(len(score))
}
