package DragonBlood

import "testing"

func TestROCArea(test *testing.T) {
	area := ROCArea([]float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
		[]bool{true, false, true, false, true, true})
	expected := 0.625

	// Coordinates are (0, 0.5), (0.5, 0.5), (0.5, 0.75), (1.0, 0.75), (1.0, 1.0)
	// Area is 0.5*0.5 + 0.5*0.75 = 1/4 + 3/8 = 0.625
	if area != expected {
		test.Errorf("ROCArea returned %g; should be %g\n", area, expected)
	}

	area = ROCArea([]float64{6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
		[]bool{true, true, false, true, false, true})

	if area != expected {
		test.Errorf("After reording, ROCArea returned %g; should be %g\n", area, expected)
	}

}

func TestMSE(test *testing.T) {
	// MSE for sequence 1, 0, 2, 3 is 2.8
	x := []float64{2.0, 2.0, 2.0, 2.0}
	y := []float64{1.0, 2.0, 4.0, 5.0}

	mse := MSE(x, y)
	expected := 3.5

	if mse != expected {
		test.Errorf("MSE returned %g; should be %g\n", mse, expected)
	}
}
