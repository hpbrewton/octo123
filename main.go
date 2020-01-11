package main

import (
	"log"
	"math"
	"math/rand" // Ayn?
	"gonum.org/v1/gonum/mat"
	"github.com/james-bowman/sparse"
)

type Brain struct {
	mag int
	stp int
	adj *sparse.Binary
	pot *mat.VecDense
	inp []int
	out []int
}

func (brain *Brain) Step() {
	brain.pot.MulVec(brain.adj, brain.pot)
	for i := 0; i < brain.mag; i++ {
		brain.pot.SetVec(i, 1.0/(1+math.Exp(-brain.pot.At(i, 0)))) // sigmoid func
	}
}

func (brain *Brain) NewBrain(mag int, pot *mat.VecDense, inp []int, adjList [][]int, out []int) {
	// building adjacency matrix
	adjData := make([]sparse.BinaryVec, mag)
	for i, _ := range adjData {
		row := sparse.NewBinaryVec(mag)
		for _, j := range adjList[i] {
			row.SetBit(j)
		}
		adjData[i] = *row
	}
	adj := sparse.NewBinary(mag, mag, adjData)

	brain.mag = mag
	brain.stp = 0
	brain.adj = adj
	brain.pot = pot
	brain.inp = inp
	brain.out = out
}

func main() {
	data := make([]float64, 10)
	for i, _ := range data {
		data[i] = rand.NormFloat64()
	}
	randvec := mat.NewVecDense(10, data)

	var brain Brain 
	brain.NewBrain(
		10,
		randvec,
		[]int{0, 1},
		[][]int{
			[]int{1, 6},
			[]int{2, 7},
			[]int{3, 8},
			[]int{4, 9},
			[]int{5, 0},
			[]int{1, 6},
			[]int{2, 7},
			[]int{3, 8},
			[]int{4, 9},
			[]int{5, 0},
		},
		[]int{8, 9},
	)
	log.Println(brain)
}