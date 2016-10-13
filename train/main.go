// Command train trains a network to find the correct
// rotation for images.
package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/unixpickle/autorot"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	DefaultInSize = 48
	BatchSize     = 32
	StepSize      = 0.001 / BatchSize
)

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) != 3 && len(os.Args) != 4 {
		dieUsage()
	}
	netFile := os.Args[1]
	imageDir := os.Args[2]
	inSize := DefaultInSize
	if len(os.Args) == 4 {
		var err error
		inSize, err = strconv.Atoi(os.Args[3])
		if err != nil {
			fmt.Fprintln(os.Stderr, "Invalid in_size:", os.Args[3])
			fmt.Fprintln(os.Stderr)
			dieUsage()
		}
	}

	network, err := autorot.LoadNetwork(netFile)
	if os.IsNotExist(err) {
		network = autorot.NewNetwork(inSize)
		log.Println("Created network.")
	} else if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load network:", err)
	} else {
		log.Println("Loaded network.")
	}

	log.Println("Reading samples...")
	samples, err := autorot.ReadSampleSet(network.InputSize, imageDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read samples:", err)
	}

	log.Println("Training...")
	cf := neuralnet.DotCost{}
	gradienter := &neuralnet.BatchRGradienter{
		Learner:       network.Net.BatchLearner(),
		CostFunc:      cf,
		MaxGoroutines: 1,
		MaxBatchSize:  BatchSize,
	}

	var iter int
	var lastBatch sgd.SampleSet
	sgd.SGDMini(gradienter, samples, StepSize, BatchSize, func(s sgd.SampleSet) bool {
		var lastCost float64
		if lastBatch != nil {
			lastCost = neuralnet.TotalCost(cf, network.Net, lastBatch)
		}
		lastBatch = s.Copy()
		cost := neuralnet.TotalCost(cf, network.Net, s)
		log.Printf("iteration %d: cost=%f last=%f", iter, cost, lastCost)
		iter++
		return true
	})

	log.Println("Saving network...")
	if err := network.Save(netFile); err != nil {
		fmt.Fprintln(os.Stderr, "Save failed:", err)
		os.Exit(1)
	}
}

func dieUsage() {
	fmt.Fprintln(os.Stderr, "Usage: train <net_file> <image_dir> [in_size]")
	os.Exit(1)
}
