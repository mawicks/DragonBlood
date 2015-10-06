package DragonBlood

import (
	"encoding/csv"
	"io"
	"os"
)

type CSVHandler interface {
	// Importer calls Header() with the first row (assume to be a header row)
	Header(header []string)

	// Importer calls Add() in each record after the required header
	// Header() will be called for Add()
	Add(data []string)

	// Importer calls Finalize() when entire file is read successfully.
	Finalize()

	// Importer calls Abort() is an error occurs before the reading the entire file.
	Abort()
}

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
	if err != nil {
		return err
	}
	return Import(file, handler)
}
