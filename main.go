package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"github.com/gonum/matrix/mat64"
	"github.com/vstoianovici/nngoclassify/neural"
	"github.com/vstoianovici/nngoclassify/pkg/config"
	"github.com/vstoianovici/nngoclassify/pkg/dataset"
)


var welcomeMsg = `
This golang Neural Network recognizes handwritten numbers form the MNIST data set (after being properly trained).

The general funtionality is the following:
 1. The first step is to initialize the NN.
 2. Read the MNIST "training" data set and train the NN.
 3. Read the MNIST "testing" data set and test the NN.
 4. Print the accuracy of the NN.
 5. Use the NN for samples outside of the "training" or "testing" datasets.
`

var (
	// path to the training data set
	traindata string
	// path to the test data set
	testdata string
	// is the data set labeled
	labeled bool
	// do we want to normalize data
	scale bool
	// manifest contains neural net config
	manifest string
)

func init() {
	flag.StringVar(&traindata, "traindata", "", "Path to training data set")
	flag.StringVar(&testdata, "testdata", "", "Path to test data set")
	flag.BoolVar(&labeled, "labeled", false, "Is the data set labeled")
	flag.BoolVar(&scale, "scale", false, "Require data scaling")
	flag.StringVar(&manifest, "manifest", "", "Path to a neural net manifest file")
}

func parseCliFlags() error {
	flag.Parse()
	// path to training data is mandatory
	if traindata == "" {
		
		return errors.New("You must specify path to training data set")
	}
	if testdata == "" {
		
		return errors.New("You must specify path to testing data set")
	}

	// path to manifest is mandatory
	if manifest == "" {
		return errors.New("You must specify path to manifest file")
	}
	return nil
}

func main() {
	fmt.Println(welcomeMsg)

	// parse cli parameters
	if err := parseCliFlags(); err != nil {
		fmt.Printf("Error parsing cli flags: %s\n", err)
		os.Exit(1)
	}
	// Read in configuration file
	config, err := config.New(manifest)
	if err != nil {
		fmt.Printf("Error reading manifest file: %s\n", err)
		os.Exit(1)
	}
	// load new training data set from provided file
	ds, err := dataset.NewDataSet(traindata, labeled)
	if err != nil {
		fmt.Printf("Unable to load Traininig Data Set: %s\n", err)
		os.Exit(1)
	}
	// extract features from data set
	features := ds.Features()
	// if we require features scaling, scale data
	if scale {
		features = dataset.Scale(features)
	}

	// extract data labels
	labels := ds.Labels()
	if labels == nil {
		fmt.Println("Data set does not contain any labels")
		os.Exit(1)
	}


	dsV, err := dataset.NewDataSet(testdata, labeled)
	if err != nil {
		fmt.Printf("Unable to load Test Data Set: %s\n", err)
		os.Exit(1)
	}
	// extract features from data set
	featuresV := dsV.Features()
	// if we require features scaling, scale data
	if scale {
		features = dataset.Scale(featuresV)
	}

	// extract data labels
	labelsV := dsV.Labels()
	if labelsV == nil {
		fmt.Println("Validation Data set does not contain any labels")
		os.Exit(1)
	}

	// Create new FEEDFWD network
	net, err := neural.NewNetwork(config.Network)
	if err != nil {
		fmt.Printf("Error creating neural network: %s\n", err)
		os.Exit(1)
	}
	// Run neural network training
	err = net.Train(config.Training, features.(*mat64.Dense), labels.(*mat64.Vector))
	if err != nil {
		fmt.Printf("Error training network: %s\n", err)
		os.Exit(1)
	}
	// check the success rate i.e. successful number of classifications
	success, err := net.Validate(featuresV.(*mat64.Dense), labelsV.(*mat64.Vector))
	if err != nil {
		fmt.Printf("Could not calculate success rate: %s\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nNeural net accuracy: %f\n", success)
	// Example of sample classification: in this case it's 1st data sample
	sample := (featuresV.(*mat64.Dense)).RowView(0).T()
	sampleLabel := int(labelsV.(*mat64.Vector).At(0,0))
	classMx, err := net.Classify(sample)
	if err != nil {
		fmt.Printf("Could not classify sample: %s\n", err)
		os.Exit(1)
	}
	fa := mat64.Formatted(classMx.T(), mat64.Prefix(""))
	fmt.Printf("\nClassification result for the first sample in dataset...\n\nFor known value(label) of the sample,  %v ...\n\n the predction vector is: \n%v\n", sampleLabel, fa)
}
