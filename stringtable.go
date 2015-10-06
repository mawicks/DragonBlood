package DragonBlood

type StringTable interface {
	// Map string to an integer and return ok if the string was
	// already in the table.
	Map(string) (result int, ok bool)

	// Return the string that maps to the passed int.
	Unmap(int) string
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
		make([]string, 0),
	}
}

func (st *stringTable) Map(s string) (int, bool) {
	m, ok := st.mapper[s]
	if !ok {
		m = len(st.unmapper)
		st.mapper[s] = m
		st.unmapper = append(st.unmapper, s)
	}
	return m, ok
}

func (st *stringTable) Unmap(i int) string {
	return st.unmapper[i]
}
