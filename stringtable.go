package DragonBlood

// StringTable is a simple encoding interface.
// Encode() maps a string to an integer and returns ok if the string was
// already in the table.  If the string was not previous known,
// Encode() still returns an integer, but returns ok == false.
// Decode(int) returns the string that maps to the passed int.
// Implementations should guarantee that Decode(Encode(string)) == string
type StringTable interface {
	Encode(string) (result int, ok bool)

	Decode(int) string
}

// stringTable is an implementation of the StringTable interface
type stringTable struct {
	mapper   map[string]int
	unmapper []string
}

// Return the canonoical StringTable implementation
func NewStringTable() *stringTable {
	return &stringTable{
		make(map[string]int),
		nil,
	}
}

func (st *stringTable) Encode(s string) (int, bool) {
	m, ok := st.mapper[s]
	if !ok {
		m = len(st.unmapper)
		st.mapper[s] = m
		st.unmapper = append(st.unmapper, s)
	}
	return m, ok
}

func (st *stringTable) Decode(i int) string {
	return st.unmapper[i]
}
