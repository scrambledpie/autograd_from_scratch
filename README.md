# autograd_from_scratch

One weekend, I wandered if I could write an autograd package using numpy and use it to train an MLP on mnist, this was the result. It is a casual, purely for fun implementation of autograd in numpy for learning/teaching purposes.

## Usage
Tensorflow is used to download the mnist dataset which is then pickled as numpy arrays.
```
python -m pip install numpy tensorflow pytest
python -m pip install -e .
pytest
python train_mnist_mlp.py
```

## Details

A handful of operations are implemented, enough for a basic Multi Layer Perceptron
- Const (compuational graph source nodes e.g. data, parameters)
- Plus
- MatMul
- Reshape
- Transpose
- Clip (Relu activation function)
- Cross-entropy Loss
- L2Distance
- SumReduce
- MaxReduce

Limitations/simplifying assumptions
- all operators assume all tensors are 2D, i.e. matrices (good enough for MLP)
- all shapes in the computational graph are constant and defined at "compile" time, before executing any compuation.
- the MNIST example does not do minibatching, the whole dataset is the minibatch.

Some nice features
- Forward pass and backward pass caching, only when a Const value is changed, all stale downstream caches are cleared
- shape checking of all tensors flowing in any direction
- hopefully easy to add new ops, the parent class Opbase has all the main functionality, individual operators need only implement 
  - `def __init__()`
  - `def _forward()`
  - `def _backward()`
  - `def shape()`
  - `def __repr__()`

Some note on implementing operators
- all operators produce only one output tensor/matrix
- if forward two inputs shaped (n,m) (k,l) and outputs (o, p), the the backward pass must be take (o, p) and produce (n,m) and (k,l)
- element-wise operators have a purely diagonal jacobian! The gradients can be computed element-wise as well