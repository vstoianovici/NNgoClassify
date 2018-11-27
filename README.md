# NNgoClassify

`NNgoClassify` provides a simple implementation of [Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network) classifier. This project was largely inspired by https://github.com/milosgajdos83/go-neural


## Get started

Get the source code:

```
$ go get -u github.com/vstoianovici/NNgoClassify
```

Build the example program:

```
$ make build

```

If the build succeeds, you should find the resulting binary in `_build` directory. Explore all of the available options:

```
$ ./_build/nnet -h
Usage of ./_build/nnet:
  -data string
        Path to training data set
  -labeled
        Is the data set labeled
  -manifest string
        Path to a neural net manifest file
  -scale
        Require data scaling
```

Run the tests:

```
$ make test
```

Feel free to explore the `Makefile` available in the root directory.

### Manifest

`NNgoClassify` allows you to define neural network architecture via a simple `YAML` file called `manifest` which can be passed to the example program shipped with the project via cli parameter. You can see the example manifest below along with some basic documentation:

```yaml
kind: feedfwd                 # network type: only feedforward networks
task: class                   # network task: only classification tasks
network:                      # network architecture: layers and activations
  input:                      # INPUT layer
    size: 784                 # 784 inputs (each input represents a pixel of a 28x28 greyscale picture representing a number)
  hidden:                     # HIDDEN layer
    size: [25]                # Array of all hidden layers
    activation: relu          # ReLU activation function
  output:                     # OUTPUT layer
    size: 10                  # 10 outputs - this implies 10 classes
    activation: softmax       # softmax activation function (excellent for classifications)
training:                     # network training
  kind: backprop              # type of training: backpropagation only
  cost: loglike               # cost function: loglikelhood (cross entropy available too)
  params:                     # training parameters
    learningrate: 0.00002     # learning rate
    epochs: 4                 # number of epochs for training
    lambda: 1.0               # lambda is a regularizer
  optimize:                   # optimization parameters
    method: bfgs              # BFGS optimization algorithm
    iterations: 80            # 80 BFGS iterations
```

As you can see the above manifest defines 3 layers neural network which uses [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function for all of its hidden layers and [softmax](https://en.wikipedia.org/wiki/Softmax_function) for its output layer. You can also specify some advanced optmization parameters. The project provides a simple manifest parser package. You can explore all available parameters in the `config` package.

### Build your own neural networks

Instead of using the manifest file and the example program provided in the root directory, you can build simple neural networks using the packages provided by the project. For example, if you want to create a simple feedforward neural network using the packages in this project, you can do so using the following code:

 ```go
package main

import (
	"fmt"
	"os"

	"github.com/vstoianovici/nngoclassify/neural"
	"github.com/vstoianovici/nngoclassify/pkg/config"
)

func main() {
	netConfig := &config.NetConfig{
		Kind: "feedfwd",
		Arch: &config.NetArch{
			Input: &config.LayerConfig{
				Kind: "input",
				Size: 100,
			},
			Hidden: []*config.LayerConfig{
				&config.LayerConfig{
					Kind: "hidden",
					Size: 25,
					NeurFn: &config.NeuronConfig{
						Activation: "sigmoid",
					},
				},
			},
			Output: &config.LayerConfig{
				Kind: "output",
				Size: 500,
				NeurFn: &config.NeuronConfig{
					Activation: "softmax",
				},
			},
		},
	}
	net, err := neural.NewNetwork(netConfig)
	if err != nil {
		fmt.Printf("Error creating network: %s\n", err)
		os.Exit(1)
	}
	fmt.Printf("Created new neural network: %v\n", net)
}
```

You can always find out more information about the functionality presented here by visiting this project's start point: https://github.com/milosgajdos83/go-neural. There you can also explore the project's packages and API in [godoc](https://godoc.org/github.com/milosgajdos83/go-neural).

## The example

The example uses the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) to train and test the neural network.

The MNIST (Modified National Institute of Standards and Technology) database contains 60,000 training images and 10,000 testing images of handwritten numbers from 0-9.

There are 2 [MNIST] data sets available in `testdata/` subdirectory to play around with, or you could download them yourself from here:

 - Training Data - [download](https://pjreddie.com/media/files/mnist_train.csv)
 - Testing Data - [download](https://pjreddie.com/media/files/mnist_test.csv)

 Furthermore, you can find multiple examples of different neural network manifest files in `manifests/` subdirectory. Fore brevit, see the results of some of the manifest configurations below.

 You can have separate runs for training, validation and prediction from a png file. Below is a run that incorporates both training and validation:


### ReLU -> Softmax -> Log Likelihood

```
time ./nnet -train ./testdata/mnist_train.csv -test ./testdata/mnist_test.csv -labeled -manifest ./manifests/example5.yml


********************************************************************************************************************************
This golang Neural Network recognizes handwritten numbers from the MNIST data set (after being properly trained).

You can either:
- perform supervised training of the network with a specified dataset
- resume/continue training provided that 1 or more epoch(s) of prior training has been performed(with previous manifest or new)
- perform validation of the trained network with a specified validation dataset
- employ the validated trained neural network to identify hand written symbols from 28x28 grayscale png files

Use the "-h" option for more details.
********************************************************************************************************************************

Training will be performed.
Testing will be performed.
No prediction will be performed outside of dataset.

--------------------------------------------------------------------------------
Started Training at: 2018-11-27 10:16:22 +0200 EET


Epoch 1...
Initial Cost: 2.440869 ... starting optimization ...
(1 out of a minimum of 400) Current Cost: 2.041604
(2 out of a minimum of 400) Current Cost: 1.649662
...
...
(407 out of a minimum of 400) Current Cost: 0.136419
(408 out of a minimum of 400) Current Cost: 0.136418
(409 out of a minimum of 400) Current Cost: 0.136418


Neural net accuracy: 97.330000

Training completed successfully at 2018-11-26 22:59:19 +0200 EET.

--------------------------------------------------------------------------------

Example (classification for the first sample in dataset):

For known value of the sample "7" ...
...the predction vector is:
⎡0.00017446823484466284⎤
⎢ 8.623852378537058e-06⎥
⎢ 0.0030112055824543515⎥
⎢   0.06511547269211937⎥
⎢ 9.093376879848648e-06⎥
⎢0.00017814626691244306⎥
⎢1.1988943512615633e-07⎥
⎢     99.92136293811964⎥
⎢  0.007095885564380252⎥
⎣ 0.0030440464209573214⎦

real	230m33.066s
user	340m20.463s
sys	45m18.598s
```

`ReLU -> Softmax -> Log Likelihood` provides the best convergence (better than cross entropy....at least in this case). 


Validation input:

./nnet -predict ../nums/0.png


Validation ouput:

```
********************************************************************************************************************************
This golang Neural Network recognizes handwritten numbers from the MNIST data set (after being properly trained).

You can either:
- perform supervised training of the network with a specified dataset
- resume/continue training provided that 1 or more epoch(s) of prior training has been performed(with previous manifest or new)
- perform validation of the trained network with a specified validation dataset
- employ the validated trained neural network to identify hand written symbols from 28x28 grayscale png files

Use the "-h" option for more details.
********************************************************************************************************************************

No training will be performed.
No testing will be performed.
Prediction based on a custom png file will be performed.

--------------------------------------------------------------------------------
Predicting at: 2018-11-27 10:13:33 +0200 EET

Printing image that will be used for prediction:


Classification output:

⎡      99.9684601829045⎤
⎢  1.87575461539393e-05⎥
⎢  0.004904835187793786⎥
⎢ 6.129118041610178e-06⎥
⎢4.0586902089434754e-05⎥
⎢0.00019896966826811355⎥
⎢   0.00262234626449708⎥
⎢ 0.0005923442479148213⎥
⎢ 3.646850780774358e-06⎥
⎣   0.02315220130996662⎦

Highest probability value: 99.9684601829045

Prediction: 0

```

There is also the possibility of resuming/continuing training by using the "-resume" argument

