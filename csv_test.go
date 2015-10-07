package DragonBlood_test

import (
	"strings"
	"testing"

	csv "github.com/mawicks/DragonBlood"
)

const (
	data string = `a,b,c
1,2,3
4,5,6`
)

var expectedHeader []string = []string{"a", "b", "c"}
var expectedRows [][]string = [][]string{{"1", "2", "3"}, {"4", "5", "6"}}

var finalizeCalled = false
var abortCalled = false

// Handler
type Handler struct {
	test   *testing.T
	header []string
	count  int
}

func (handler *Handler) Header(header []string) {
	handler.header = header
}

func (handler *Handler) Add(fields []string) {
	for i, _ := range expectedHeader {
		if expectedRows[handler.count][i] != fields[i] {
			handler.test.Errorf("Got %v; expected %v",
				fields[i], expectedRows[handler.count][i])
		}
	}
	handler.count += 1
}

func (handler *Handler) Finalize() {
	finalizeCalled = true

	for i, h := range handler.header {
		if expectedHeader[i] != h {
			handler.test.Errorf("Got %v; expected %v", h, expectedHeader[i])
		}
	}
}

func (handler *Handler) Abort() {
	abortCalled = true
}

// TestCSV
func TestCSV(test *testing.T) {
	reader := strings.NewReader(data)
	err := csv.Import(reader, &Handler{test, nil, 0})

	if err != nil {
		test.Error("Import returned error")
	}

	if !finalizeCalled {
		test.Error("Finalize was never called")
	}

	if abortCalled {
		test.Error("Abort should not have been called")
	}
}
