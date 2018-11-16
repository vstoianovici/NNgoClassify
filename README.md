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

	"github.com/vstoianovici/nngolassify/neural"
	"github.com/vstoianovici/nngolassify/pkg/config"
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


### ReLU -> Softmax -> Log Likelihood

```
time ./nnet -data ../testdata/mnist_test.csv -labeled -manifest ../manifests/example5.yml
Current Cost: 2.432839
Current Cost: 2.018512
Current Cost: 1.625155
...
...
Current Cost: 0.027839
Current Cost: 0.026826
Current Cost: 0.026287
Result status: IterationLimit

Neural net accuracy: 99.500000

Classification result:
⎡ 5.298480689850579e-07⎤
⎢ 7.952260749764308e-18⎥
⎢ 1.826173561323361e-10⎥
⎢ 0.0012653529377857982⎥
⎢ 2.423682324787905e-18⎥
⎢2.0833074305356955e-12⎥
⎢ 1.767464983293055e-24⎥
⎢     99.99873411701802⎥
⎢ 3.341523095120271e-15⎥
⎣1.1405043387000189e-11⎦


real	4m24.000s
user	5m25.834s
sys	0m31.748s
```

`ReLU -> Softmax -> Log Likelihood` provides the best convergence (better than cross entropy....at least in this case). Right now I am using the training data for validation as well, but at a certain point I will change the validation to use a separate dataset.
