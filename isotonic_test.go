package DragonBlood_test

import (
	"testing"

	db "github.com/mawicks/DragonBlood"
)

// RocArea computes the area under the ROC curve
// score is any score that orders the data (e.g. a probability estimate)
// target contains the labels (True == positive; False == negative)
func TestIsotonic(test *testing.T) {
	x, y := db.Isotonic(
		[]float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
		[]float64{1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0})

	expectedX := []float64{1.0, 3.0, 7.0}
	expectedY := []float64{0.5, 0.75, 5.0 / 6.0}

	for i := range x {
		if x[i] != expectedX[i] {
			test.Errorf("x[%d] was %g; expected %g", i, x[i], expectedX[i])
		}
		if y[i] != expectedY[i] {
			test.Errorf("y[%d] was %g; expected %g", i, y[i], expectedY[i])
		}
	}

	x, y = db.Isotonic(
		[]float64{12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
		[]float64{0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0})

	for i := range x {
		if x[i] != expectedX[i] {
			test.Errorf("x[%d] was %g; expected %g", i, x[i], expectedX[i])
		}
		if y[i] != expectedY[i] {
			test.Errorf("y[%d] was %g; expected %g", i, y[i], expectedY[i])
		}
	}
}
