package DragonBlood_test

import (
	"testing"

	db "github.com/mawicks/DragonBlood"
)

// TestStringTable
func TestStringTable(test *testing.T) {
	test_strings := []string{"a", "b", "c", "d", "e"}

	table := db.NewStringTable()

	// Map every string twice.  During the first cycle, the string
	// should not be found.  During the second cycle it should be
	// found.  Unmap should return the original value during both
	// cycles.
	for i := 0; i < 2; i++ {
		for _, s := range test_strings {
			m, ok := table.Map(s)
			u := table.Unmap(m)
			if u != s {
				test.Errorf(`"%s" mapped back to "%s" during cycle %d`, s, u, i)
			}
			if ok && i == 0 {
				test.Errorf(`"%s" found during fill cycle`, s)
			}
			if !ok && i == 1 {
				test.Errorf(`"%s" not found during verify cycle`, s)
			}
		}
	}
}
