package DragonBlood

import "testing"

func TestROCArea(test *testing.T) {
	area := ROCArea([]float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
		[]bool{true, false, true, false, true, true})

	// Coordinates are (0, 0.5), (0.5, 0.5), (0.5, 0.75), (1.0, 0.75), (1.0, 1.0)
	// Area is 0.5*0.5 + 0.5*0.75 = 1/4 + 3/8 = 0.625
	if area != 0.625 {
		test.Errorf("ROCArea returned %g; should be %g\n", area, 0.625)
	}

	area = ROCArea([]float64{6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
		[]bool{true, true, false, true, false, true})

	if area != 0.625 {
		test.Errorf("After reording, ROCArea returned %g; should be %g\n", area, 0.625)
	}

}
