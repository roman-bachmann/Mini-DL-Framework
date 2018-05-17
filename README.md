# Mini-DL-Framework
Second project of the EPFL Spring 2018 Deep Learning Class

## Dependencies

- Python 3.5.4
- PyTorch 3.0.1
- Matplotlib 2.0.2

## Folder structure

- **src/**: Path to all the source code
	- **activations.py**: Contains ReLU, Tanh and Sigmoid activation function modules
	- **containers.py**: Contains Sequential container for building sequential neural networks
	- **initializers.py**: Contains Xavier normal, Xavier uniform and standard uniform initializers
	- **layers.py**: Contains Linear and Dropout layer modules
	- **losses.py**: Contains MSE loss function module
	- **module.py**: Abstract module class
	- **optimizers.py**: Contains SGD, RMSProp and Adam optimizers
	- **parameter.py**: Contains parameter container holding weights and gradients
	- **test.py**: Main test executable running a neural network with 3 hidden layers to classify if a point is inside a circle or not
	- **test\_dropout.py**: Benchmarking an overfitting model with and without Dropout layers
	- **test\_optimizers.py**: Benchmarking convergence of different optimizers
	- **test\_speed.py**: Benchmarking training speed of a given network with our framework against PyTorch
- **README.md**

## How to run test files

To run the main executable, run:

```
$ cd src && python test.py
```