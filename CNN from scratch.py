import numpy as np
import matplotlib as plt

np.random.seed(0)  # For reproductibility


### Layers

## Mother  class

class layer :

    def __init__(self) :

        self.input = None
        self.output = None

    def forward(self, input) :
        pass

    def backward(self, grad, eta) :    # eta : learning rate
        pass

## Dense layer

class dense_layer(layer) :

    def __init__(self, nb_inputs, nb_neurones) :

        self.weights  = np.random.randn(nb_neurones, nb_inputs)
        self.biases = np.random.randn(nb_neurones, 1)

    def forward(self, input) :

        self.input = input

        return np.dot(self.weights, self.input) + self.biases

    def backward(self, grad, eta) :

        weights_grad = np.dot(grad, self.input.T)

        self.weights -= eta*weights_grad      # weights update
        self.biases -= eta*grad             # biases update

        return np.dot(self.weights.T, grad)

## Activation layer

class activation_layer(layer) :

    def __init__(self, activ_func, derivative) :

        self.activ_func = activ_func
        self.derivative = derivative


    def forward(self, input) :

        self.input = input

        return self.active_func(self.input)

    def backward(self, grad, eta) :

        return np.multiply(grad, self.derivative(self.input))

# sigmoid activation

class sigmoid_activation_layer(activation_layer) :

    def __init_(self) :

        sigmoid = lambda x : 1/(1 + np.exp(-x))
        sigmoid_derivative = lambda x : 1/(1 + np.exp(-x))*(1 - 1/(1 + np.exp(-x)))
        super().__init__(sigmoid, sigmoid_derivative)


## Convolution  layer

from scipy import signal

class convolutionnal_layer(layer) :

    def __init__(self, input_shape, kernel_size, depth) :


        # We first deal with the dimension of our images, kernes and features

        input_depth, input_height, input_width =  input_shape

        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        # We iniatlize the kernels and the biases

        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input) :

        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth) :
            for j in range(self.input_depth) :

                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], 'valid')

        return self.ouptut

    def backward(self, grad, eta) :

        kernels_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeros(self.input_shape)

        for i in range(self.depth) :
            for j in range(self.input_depth) :

                kernels_grad[i,j] = signal.correlate2d(self.input[j], grad[i], 'valid')
                input_grad[j] +=  signal.convolve2d(grad[i], self.kernels[i,j], 'full')

        self.kernels -= eta * kernels_grad
        self.biases -= eta * grad

        return input_grad

## Reshape_layer


class reshape_layer(layer) :

    def __init__(self, input_shape, output_shape) :

        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input) :

        return np.reshape(input, self.output_shape)

    def backward(self, grad, eta) :

        return np.reshape(grad, self.input_shape)

### Test ###
















