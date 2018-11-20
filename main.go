package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"time"
	"github.com/gonum/matrix/mat64"
	"github.com/vstoianovici/nngoclassify/neural"
	"github.com/vstoianovici/nngoclassify/pkg/config"
	"github.com/vstoianovici/nngoclassify/pkg/dataset"
)


var welcomeMsg = `
****************************************************************************************************************************
This golang Neural Network recognizes handwritten numbers from the MNIST data set (after being properly trained).

You can either:
- perform supervised training of the network with a specified dataset
- perform validation of the trained network with a specified validation dataset
- employ the validated trained neural network to identify hand written symbols from 28x28 grayscale png files

Use the "-h" option for more details.
****************************************************************************************************************************
`

var (
	// path to the training data set
	train string
	// path to the test data set
	test string
	// path to png picture to predict
	predict string
	// is the data set labeled
	labeled bool
	// do we want to normalize data
	scale bool
	// manifest contains neural net config
	manifest string

	isTraining bool
	isTesting bool
	isPredicting bool
)

func init() {
	flag.StringVar(&train, "train", "", "Path to training data set")
	flag.StringVar(&test, "test", "", "Path to test data set")
	flag.StringVar(&test, "predict", "", "Path to png file used for prediction")
	flag.BoolVar(&labeled, "labeled", false, "Is the data set labeled")
	flag.BoolVar(&scale, "scale", false, "Require data scaling")
	flag.StringVar(&manifest, "manifest", "", "Path to the neural net manifest file")
}

func parseCliFlags() error {
	flag.Parse()
	// path to training data is mandatory
	if train == "" {	
		fmt.Println("No training will be performed.")
		isTraining = false
		//return errors.New("You must specify path to training data set")
		if test == "" {
			fmt.Println("No testing will be performed.")
			isTesting = false
			if predict == "" {
				fmt.Println("No prediction will be performed outside of dataset.\n")
				isPredicting = false
				return errors.New("No action was specified. At least one action needs to performed (train, test or predict).")
			}else{
				fmt.Println("Prediction based on a custom png file will be performed.\n")
				isPredicting = true
			}

		}else{
			fmt.Println("Testing will be performed.")
			isTesting = true
			if predict == "" {
				fmt.Println("No prediction will be performed outside of dataset.\n")
				isPredicting = false
			}else{
				fmt.Println("Prediction based on a custom png file will be performed.\n")
				isPredicting = true
			}
		}
	}else{
		isTraining = true
		fmt.Println("Training will be performed.")
		// path to manifest is mandatory
		if manifest == "" {
			return errors.New("You must specify path to manifest file")
		}
		if test == "" {
			fmt.Println("No testing will be performed.")
			isTesting = false
		}else{
			fmt.Println("Testing will be performed.")
			isTesting = true
		}
		if predict == "" {
				fmt.Println("No prediction will be performed outside of dataset.\n")
				isPredicting = false
		}else{
				fmt.Println("Prediction based on a custom png file will be performed.\n")
				isPredicting = true
		}
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
	
	var net *neural.Network

	if isTraining {
		fmt.Println("--------------------------------------------------------------------------------")
		secs := time.Now().Unix()
		fmt.Printf("Started Training at: %s\n\n", time.Unix(secs, 0))
		// Read in configuration file
		configuration, err := config.New(manifest)
		if err != nil {
			fmt.Printf("Error reading manifest file: %s\n", err)
			os.Exit(1)
		}	
	
		// load new training data set from provided file
		ds, err := dataset.NewDataSet(train, labeled)
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
		//fmt.Println(mat64.Formatted(features))

		// extract data labels
		labels := ds.Labels()
		if labels == nil {
			fmt.Println("Data set does not contain any labels")
			os.Exit(1)
		}

		// Create new FEEDFWD network
		net, err = neural.NewNetwork(configuration.Network)
		if err != nil {
			fmt.Printf("Error creating neural network: %s\n", err)
			os.Exit(1)
		}

		// Run neural network training
		err = net.Train(configuration.Training, features.(*mat64.Dense), labels.(*mat64.Vector), manifest)
		if err != nil {
			fmt.Printf("Error training network: %s\n", err)
			os.Exit(1)
		}
		secs = time.Now().Unix()
		fmt.Printf("\nTraining completed successfully at %s.\n\n", time.Unix(secs, 0))

	}
	
	if isTesting {
		fmt.Println("--------------------------------------------------------------------------------")
		secs := time.Now().Unix()
		fmt.Printf("Started Testing/Validation at: %s\n", time.Unix(secs, 0))
		if !isTraining {
			fmt.Println("We need to recreate the network from stuff that's already in /trainingdata")
			//process the manifest saved during training
			configuration, err := config.New("trainingdata/trainedManifest.yml")
			if err != nil {
				fmt.Printf("Error reading manifest file: %s\n", err)
				os.Exit(1)
			}
			// Recreate FEEDFWD network from saved manifest
			net, err = neural.NewNetwork(configuration.Network)
			if err != nil {
				fmt.Printf("Error creating neural network: %s\n", err)
				os.Exit(1)
			}
			neural.LoadFromFile(net)
		}

		dsV, err := dataset.NewDataSet(test, labeled)
		if err != nil {
			fmt.Printf("Unable to load Test Data Set: %s \n\n", err)
			os.Exit(1)
		}
		// extract features from data set
		featuresV := dsV.Features()
		// if we require features scaling, scale data
		if scale {
			featuresV = dataset.Scale(featuresV)
		}

		// extract data labels
		labelsV := dsV.Labels()
		if labelsV == nil {
			fmt.Println("Validation Data set does not contain any labels")
			os.Exit(1)
		}

		// check the success rate i.e. successful number of classifications
		success, err := net.Validate(featuresV.(*mat64.Dense), labelsV.(*mat64.Vector))
		if err != nil {
			fmt.Printf("Could not calculate success rate: %s\n", err)
			os.Exit(1)
		}
		fmt.Printf("\nNeural net accuracy: %f\n", success)

		secs = time.Now().Unix()
		fmt.Printf("\nTesting/Validation completed successfully at %s.\n", time.Unix(secs, 0))
		// Example of sample classification: in this case it's 1st data sample
		sample := (featuresV.(*mat64.Dense)).RowView(0).T()
		sampleLabel := int(labelsV.(*mat64.Vector).At(0,0))
		classMx, err := net.Classify(sample)
		if err != nil {
			fmt.Printf("Could not classify sample: %s\n", err)
			os.Exit(1)
		}
		fa := mat64.Formatted(classMx.T(), mat64.Prefix(""))
		fmt.Println("--------------------------------------------------------------------------------")
		fmt.Printf("\nExample (classification for the first sample in dataset):\n\nFor known value of the sample \"%v\" ...\n...the predction vector is: \n%v\n", sampleLabel, fa)
	}
}
