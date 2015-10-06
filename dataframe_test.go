package DragonBlood_test

import (
	"fmt"
	"testing"

	db "github.com/mawicks/DragonBlood"
)

func TestDataFrame(t *testing.T) {
	names := []string{"a", "b", "c", "d"}

	data := [][]interface{}{
		{"1.0", "2.0", "3.0", "4.0"},
		{1.1, 2.2, 3.3, 4.4}}

	expected := [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{1.1, 2.2, 3.3, 4.4}}

	nf := db.NewDataFrame()
	for i := 0; i < 4; i++ {
		nf.AddFeature(db.NewNumericFeature(names[i]))
	}

	if nf.Length() != 0 {
		t.Error("Length should be 0")
	}

	for _, v := range data {
		nf.AddRow(v)
	}

	if nf.Length() != 2 {
		t.Error("Length should be 2")
	}

	if nf.Width() != 4 {
		t.Error("Columns() should return 4")
	}

	for i := 0; i < nf.Length(); i++ {
		for j := 0; j < nf.Width(); j++ {
			if nf.Get(i, j) != expected[i][j] {
				t.Error(fmt.Sprintf("Cell (%d, %d): got %v; expected %v",
					i, j, nf.Get(i, j), expected[i][j]))
			}
		}
	}

	for j := 0; j < nf.Width(); j++ {
		if nf.ColumnName(j) != names[j] {
			fmt.Println(nf.ColumnName(j))
			t.Error(fmt.Sprintf(`Column name: got "%v"; expected "%v"`, nf.ColumnName(j), names[j]))
		}
	}
}
