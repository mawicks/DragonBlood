package DragonBlood_test

import (
	"fmt"
	"strings"
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

func featureFactory(s string) db.Feature {
	switch s[0] {
	case 'C':
		return db.NewCategoricalFeature(s, db.NewStringTable())
	default:
		return db.NewNumericFeature(s)
	}
}

func TestCSVHandler(t *testing.T) {
	reader := strings.NewReader(`apple,baker,Charlie
1,"x","foo"
4,"5","baz"`)
	handler := db.NewCSVHandler(featureFactory)

	if db.Import(reader, handler) != nil {
		t.Error("Import() returned an error")
	}

	df := handler.DataFrame()
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			fmt.Printf("%v ", df.Get(i, j))
		}
		fmt.Printf("\n")
	}
}
