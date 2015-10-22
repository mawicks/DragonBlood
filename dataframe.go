package DragonBlood

import "fmt"

type DataFrameFeature interface {
	Feature
	Name() string
}

type dataFrameFeature struct {
	Feature
	name string
}

func (dff *dataFrameFeature) Name() string {
	return dff.name
}

func NewDataFrameFeature(name string, feature Feature) *dataFrameFeature {
	return &dataFrameFeature{feature, name}
}

type DataFrame struct {
	feature   []DataFrameFeature
	columnMap map[string]int
	length    int
}

func NewDataFrame() *DataFrame {
	return &DataFrame{nil,
		make(map[string]int),
		0,
	}
}

func (df *DataFrame) AddFeature(feature DataFrameFeature) {
	featureLength := feature.Len()
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
			if f.Len() == df.length {
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

func (df *DataFrame) AddStringRow(row []string) {
	if len(df.feature) == len(row) {
		for i, f := range df.feature {
			if f.Len() == df.length {
				f.AddFromString(row[i])
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
		result[i] = f.Value(index)
	}

	return result
}

func (df *DataFrame) Get(i, j int) interface{} {
	return df.feature[j].Value(i)
}

type csvHandler struct {
	featureFactory func(string) DataFrameFeature
	df             *DataFrame
}

func (handler *csvHandler) Header(header []string) {
	for _, h := range header {
		f := handler.featureFactory(h)
		handler.df.AddFeature(f)
	}
}

func (handler *csvHandler) Add(row []string) {
	handler.df.AddStringRow(row)
}

func (handler *csvHandler) Finalize()             {}
func (handler *csvHandler) Abort()                {}
func (handler *csvHandler) DataFrame() *DataFrame { return handler.df }

func NewCSVHandler(featureFactory func(string) DataFrameFeature) *csvHandler {
	return &csvHandler{featureFactory, NewDataFrame()}
}

func CSVFileToDataFrame(filename string, featureFactory func(header string) DataFrameFeature) error {
	return ImportFile(filename, NewCSVHandler(featureFactory))
}
