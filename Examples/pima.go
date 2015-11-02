package main

import (
	"fmt"
	"io"
	"log"
	"os"

	db "github.com/mawicks/DragonBlood"
	"github.com/mawicks/DragonBlood/stats"
)

// Handler
type Handler struct {
	header  []string
	columns []*db.NumericFeature

	features []db.OrderedFeature
	target   *db.NumericFeature
}

func (handler *Handler) Header(header []string) {
	handler.header = header
	handler.columns = make([]*db.NumericFeature, len(header))
	for i := range handler.columns {
		handler.columns[i] = db.NewNumericFeature()
	}
}

func (handler *Handler) Add(fields []string) {
	for i, f := range fields {
		handler.columns[i].AddFromString(f)
	}
}

func (handler *Handler) Finalize() {
	handler.features = []db.OrderedFeature{}

	for i, h := range handler.header {
		if h[0] == 'f' {
			handler.features = append(handler.features, handler.columns[i])
		} else if h[0] == 't' {
			handler.target = handler.columns[i]
		}
	}

}

func (handler *Handler) Abort() {
}

// TestCSV
func main() {
	var reader io.Reader
	var err error
	if reader, err = os.Open("pima-indians-diabetes.csv"); err != nil {
		log.Fatalf("%v", err)
	}

	handler := &Handler{}
	err = db.Import(reader, handler)

	targetBool := make([]bool, handler.target.Len())
	targetNumeric := make([]float64, handler.target.Len())
	for i := 0; i < handler.target.Len(); i++ {
		targetBool[i] = handler.target.NumericValue(i) == 1.0
		targetNumeric[i] = handler.target.NumericValue(i)
	}

	if err != nil {
		log.Fatal("Import returned error")
	}

	rf := db.NewRandomForest(100).SetMaxFeatures(5).SetMinLeafSize(29)
	//	mf := db.NewEntropyCriterionFactory()
	oobScores := rf.Fit(handler.features, handler.target, db.NewMSECriterionFactory())

	auc := db.ROCArea(oobScores, targetBool)
	mse := db.MSE(oobScores, targetNumeric)
	variance := stats.Variance(targetNumeric)

	importances := rf.Importances()

	oobGini := rf.Fit(handler.features, handler.target, db.NewGiniCriterionFactory(handler.target.Len()))
	accuracyGini := db.Accuracy(oobGini, targetNumeric)

	fmt.Printf("importances: %v\n", importances)
	//	fmt.Printf("oob(gini): %v\n", oobGini)
	//	fmt.Printf("target: %v\n", targetNumeric)

	fmt.Printf("ROCArea: %v MSE: %v Var: %v\n", auc, mse, variance)
	fmt.Printf("Accuracy(gini): %v\n", accuracyGini)
}
