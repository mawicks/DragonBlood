package DragonBlood_test

import (
	"testing"

	db "github.com/mawicks/DragonBlood"
)

func TestBag(test *testing.T) {
	for _, bag := range []db.Bag{db.NewBag(10), db.FullBag(10)} {
		sum := 0
		for i := 0; i < bag.Len(); i++ {
			sum += bag.Count(i)
		}
		if sum != bag.Len() {
			test.Errorf("%+v: Sum of bag counts (%d)  doesn't equal its length (%d)", bag, sum, bag.Len())
		}
	}
}
