package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	pprof "runtime/pprof"

	db "github.com/mawicks/DragonBlood"
)

// Handler
type Handler struct {
	featuremap map[string]int
	header     []string

	features []db.OrderedFeature
	target   *db.CategoricalFeature
	count    int
}

func (handler *Handler) Header(header []string) {
	handler.header = header
	handler.features = []db.OrderedFeature{}
	for _, h := range header {
		if h[0] == 'p' {
			handler.featuremap[h] = len(handler.features)
			handler.features = append(handler.features, db.NewNumericFeature())
		}
		if h[0] == 'l' {
			handler.target = db.NewCategoricalFeature()
		}
	}
}

func (handler *Handler) Add(fields []string) {
	if handler.count < 100000 {
		for i, f := range fields {
			if handler.header[i][0] == 'p' {
				handler.features[handler.featuremap[handler.header[i]]].AddFromString(f)
			}
			if handler.header[i][0] == 'l' {
				handler.target.AddFromString(f)
			}
		}
	}
	handler.count += 1
}

func (handler *Handler) Finalize() {}

func (handler *Handler) Abort() {
}

var cpuprofile = flag.String("cpuprofile", "", "Write cpu profile to file")

// TestCSV
func main() {
	var reader io.Reader
	var err error

	flag.Parse()

	if reader, err = os.Open("digits-train.csv"); err != nil {
		log.Fatalf("%v", err)
	}

	handler := &Handler{featuremap: make(map[string]int)}
	err = db.Import(reader, handler)

	targetNumeric := make([]float64, handler.target.Len())
	for i := 0; i < handler.target.Len(); i++ {
		targetNumeric[i] = handler.target.NumericValue(i)
	}

	if err != nil {
		log.Fatal("Import returned error")
	}

	fmt.Printf("target: %v\n", targetNumeric)

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}

		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	rf := db.NewRandomForest(10).SetMaxFeatures(30).SetMinLeafSize(1)

	for h, i := range handler.featuremap {
		fmt.Printf("%s: %d  ", h, handler.features[i].Len())
	}

	fmt.Printf("target len: %d range: %d\n", handler.target.Len(), handler.target.Range())

	oobGini := rf.Fit(handler.features, handler.target, db.NewGiniCriterionFactory(handler.target.Range()))
	accuracyGini := db.Accuracy(oobGini, targetNumeric)

	//	importances := rf.Importances()

	//	fmt.Printf("importances: %v\n", importances)
	//	fmt.Printf("%v", oobGini)

	fmt.Printf("Accuracy(gini): %v\n", accuracyGini)
}
