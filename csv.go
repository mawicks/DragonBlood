package DragonBlood

import (
	"encoding/csv"
	"io"
	"os"
)

type CSVHandler interface {
	// Import() calls CSVHandler.Header() with the first row (assumed to be a header row)
	Header(header []string)

	// Import() calls CSVHandler.Add() in each record after the required header
	// Header() will be called for Add()
	Add(data []string)

	// Import() calls CSVHandler.Finalize() when entire file is read successfully.
	Finalize()

	// Import() calls Abort() if any error occurs before reading to the end of the file.
	Abort()
}

// Import() reads a CSV file from an io.Reader object.
// Client-provided callbacks are provided by an implemention of
// CSVHandler.  It returns any error that occured file reading or
// parsing the file.
func Import(reader io.Reader, handler CSVHandler) error {
	var data []string
	csvReader := csv.NewReader(reader)
	header, err := csvReader.Read()

	if err == nil {
		handler.Header(header)
		for data, err = csvReader.Read(); err == nil; data, err = csvReader.Read() {
			handler.Add(data)
		}
	}

	if err == io.EOF {
		handler.Finalize()
		return nil
	} else {
		handler.Abort()
		return err
	}
}

func ImportFile(filename string, handler CSVHandler) error {
	file, err := os.Open(filename)
	if err == nil {
		defer file.Close()
		return Import(file, handler)
	} else {
		return err
	}
}
