package DragonBlood

import "fmt"

type DataFrame struct {
	feature   []Feature
	columnMap map[string]int
	empty     bool
}

func NewDataFrame() *DataFrame {
	return &DataFrame{make([]Feature, 0),
		make(map[string]int),
		true}
}

func (nf *DataFrame) AddFeature(feature Feature) {
	if nf.empty {
		nf.columnMap[feature.Name()] = len(nf.feature)
		nf.feature = append(nf.feature, feature)
	} else {
		panic("Attempt to add an empty column to a non-empty dataframe")
	}
}

func (nf *DataFrame) AddRow(row []interface{}) {
	if len(nf.feature) == len(row) {
		nf.empty = false
		for i, value := range row {
			nf.feature[i].Add(value)
		}
	} else {
		panic(fmt.Sprintf("Attempt to add a row of length %d to "+
			"DataFrame with %d columns", len(row), len(nf.feature)))
	}
}

func (nf *DataFrame) Width() int { return len(nf.feature) }

func (nf *DataFrame) Length() int {
	if nf.empty {
		return 0
	} else {
		return nf.feature[0].Length()
	}
}

func (nf *DataFrame) ColumnName(index int) string {
	return nf.feature[index].Name()
}

func (nf *DataFrame) Row(index int) []interface{} {
	result := make([]interface{}, len(nf.feature))

	for i, f := range nf.feature {
		result[i] = f.Get(index)
	}

	return result
}

func (nf *DataFrame) Get(i, j int) interface{} {
	return nf.feature[j].Get(i)
}
