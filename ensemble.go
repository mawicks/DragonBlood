package DragonBlood

import "math/rand"

// Bag
type Bag interface {
	Len() int
	Count(int) int
	Resample()
}

type bag []int

func NewBag(n int) bag {
	newBag := bag(make([]int, n))
	newBag.Resample()
	return newBag
}

func (b bag) Resample() {
	for i, _ := range b {
		b[i] = 0
	}

	n := len(b)
	for i := 0; i < n; i++ {
		b[rand.Intn(n)] += 1
	}
}

func (b bag) Count(i int) int { return b[i] }

func (b bag) Len() int { return len(b) }

type FullBag int

func (n FullBag) Len() int     { return int(n) }
func (FullBag) Resample()      {}
func (FullBag) Count(int) int  { return 1 }
func (FullBag) String() string { return "Full Bag (all samples)" }
