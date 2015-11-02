package DragonBlood

// Codec is a simple encoding interface.
// Encode() maps a string to an integer and returns ok if the string was
// already in the table.  If the string was not previous known,
// Encode() still returns an integer, but returns ok == false.
// Decode(int) returns the string that maps to the passed int.
// Implementations should guarantee that Decode(Encode(string)) == string
type Codec interface {
	Encode(interface{}) (result int, ok bool)
	Decode(int) interface{}
	Len() int
}

// codec is an implementation of the Codec interface
type codec struct {
	mapper   map[interface{}]int
	unmapper []interface{}
}

// Return the canonoical Codec implementation
func NewCodec() *codec {
	return &codec{
		make(map[interface{}]int),
		nil,
	}
}

func (c *codec) Encode(s interface{}) (int, bool) {
	m, ok := c.mapper[s]
	if !ok {
		m = len(c.unmapper)
		c.mapper[s] = m
		c.unmapper = append(c.unmapper, s)
	}
	return m, ok
}

func (c *codec) Decode(i int) interface{} {
	return c.unmapper[i]
}

func (c *codec) Len() int {
	return len(c.unmapper)
}
