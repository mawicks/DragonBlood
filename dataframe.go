package DragonBlood

import "fmt"

type DataFrame struct {
	feature   []Feature
	columnMap map[string]int
	length    int
}

func NewDataFrame() *DataFrame {
	return &DataFrame{make([]Feature, 0),
		make(map[string]int),
		0,
	}
}

func (df *DataFrame) AddFeature(feature Feature) {
	featureLength := feature.Length()
	if df.length == 0 || featureLength == df.length {
		df.columnMap[feature.Name()] = len(df.feature)
		df.feature = append(df.feature, feature)
		df.length = featureLength
	} else {
		panic("Attempt to add column with mismatched length to a non-empty dataframe")
	}
}

func (df *DataFrame) AddRow(row []interface{}) {
	if len(df.feature) == len(row) {
		for i, f := range df.feature {
			if f.Length() == df.length {
				f.Add(row[i])
			} else {
				panic("Attempt to add row to dataframe with mismatched feature lengths")
			}
		}
		df.length += 1
	} else {
		panic(fmt.Sprintf("Attempt to add a row of length %d to "+
			"DataFrame with %d columns", len(row), len(df.feature)))
	}
}

func (df *DataFrame) Width() int { return len(df.feature) }

func (df *DataFrame) Length() int {
	return df.length
}

func (df *DataFrame) ColumnName(index int) string {
	return df.feature[index].Name()
}

func (df *DataFrame) Row(index int) []interface{} {
	result := make([]interface{}, len(df.feature))

	for i, f := range df.feature {
		result[i] = f.Get(index)
	}

	return result
}

func (df *DataFrame) Get(i, j int) interface{} {
	return df.feature[j].Get(i)
}
