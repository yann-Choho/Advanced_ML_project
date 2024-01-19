### Imports ###

import numpy as np
from scipy import signal

np.random.seed(0)  # For reproductibility

### Mother class ###

class layer :

    def __init__(self) :

        self.input = None
        self.output = None

    def forward(self, input) :
        pass

    def backward(self, grad, eta) :    # eta : learning rate
        pass

### Dense layer ###

class dense_layer(layer) :

    def __init__(self, nb_inputs, nb_neurones) :

        # We will use the Xavier initializer

        self.weights  = np.sqrt(2/nb_inputs)*np.random.randn(nb_neurones, nb_inputs)
        self.biases = np.zeros((nb_neurones,1))

    def forward(self, input) :

        self.input = input

        return np.dot(self.weights, self.input) + self.biases

    def backward(self, grad, eta) :

        weights_grad = np.dot(grad, self.input.T)

        self.weights -= eta*weights_grad      # weights update
        self.biases -= eta*grad             # biases update


        return np.dot(self.weights.T, grad)

### Convolutionnal layer ###

class convolutionnal_layer(layer) :

    def __init__(self, input_shape, kernel_size, depth) :


        # We first deal with the dimension of our images, kernes and features

        input_depth, input_height, input_width =  input_shape

        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        # We iniatlize the kernels with the Xavier-Glorot method

        # fan = input_width + self.output_shape[1]
        # self.kernels = np.zeros(self.kernels_shape)

        # for i in range(self.kernels_shape[0]) :
        #     for j in range(self.kernels_shape[1]) :
        #         for k in range(self.kernels_shape[2]) :
        #             for l in range(self.kernels_shape[3]) :

        #                 self.kernels[i][j][k][l] = np.random.uniform(low = -np.sqrt(6/fan), high = np.sqrt(6/fan))

        # self.biases = np.zeros(self.output_shape)

        # He initialization for ReLU activation
        output_size = (depth*( input_height - kernel_size + 1))**2
        self.kernels = np.sqrt(2/(depth*(kernel_size**2 + output_size)))*np.random.randn(*self.kernels_shape)
        self.biases = np.sqrt(2/(depth*(kernel_size**2 + output_size)))*np.random.randn(*self.output_shape)

    def forward(self, input) :

        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth) :
            for j in range(self.input_depth) :

                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], 'valid')

        return self.output

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


### Activation layers ###

class sigmoid_activation_layer(layer) :

    def __init__(self) :

        self.activ_func = lambda x : 1/(1 + np.exp(-x))
        self.derivative = lambda x : 1/(1 + np.exp(-x))*(1 - 1/(1 + np.exp(-x)))

    def forward(self, input) :

        self.input = input
        return self.activ_func(self.input)

    def backward(self, grad, eta) :

        return np.multiply(grad, self.derivative(self.input))

class softmax_activation_layer(layer) :

    def forward(self,input) :

        self.output = np.exp(input)/np.sum(np.exp(input))

        return np.clip(self.output, 10e-7, 1 - 10e-7)

    def backward(self,grad,eta) :

        return np.dot(grad, self.output)*self.output + np.multiply(grad,self.output)

class tanh_activation_layer(layer) :

    def __init__(self) :

        self.activ_func = lambda x : np.tanh(x)
        self.derivative = lambda x : 1 - np.tanh(x)**2

    def forward(self, input) :

        self.input = input
        return self.activ_func(self.input)

    def backward(self, grad, eta) :

        return np.multiply(grad, self.derivative(self.input))

class ReLU_activation_layer(layer) :

    def __init__(self) :

        self.activ_func = lambda x : np.maximum(0,x)

        def relu_derivative(x) :

            x[x<=0] = 0
            x[x>0] = 1

            return x

        self.derivative = relu_derivative

    def forward(self, input) :

        self.input = input
        return self.activ_func(self.input)

    def backward(self, grad, eta) :

        return np.multiply(grad, self.derivative(self.input))

### Reshape layer ###

class reshape_layer(layer) :

    def __init__(self, input_shape, output_shape) :

        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input) :

        return np.reshape(input, self.output_shape)

    def backward(self, grad, eta) :

        return np.reshape(grad, self.input_shape)

### Pooling layer ###

class avg_pool_layer(layer) :

    def __init__(self, input_shape, kernel_size) :


        # We first deal with the dimension of our images, kernes and features

        input_depth, input_height, input_width =  input_shape

        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (input_depth, input_height - kernel_size + 1, input_width - kernel_size + 1)

        # We iniatlize the kernel

        self.kernels = kernel_size**(-1)*np.ones((input_depth,kernel_size,kernel_size))

    def forward(self, input) :

        self.input = input
        self.output = np.zeros(self.output_shape)

        for i in range(self.input_depth) :

            self.output[i] += signal.correlate2d(self.input[i], self.kernels[i], 'valid')

        return self.output

    def backward(self, grad, eta) :

        input_grad = np.zeros(self.input_shape)

        for i in range(self.input_depth) :

            input_grad[i] +=  signal.convolve2d(grad[i], self.kernels[i], 'full')

        return input_grad

### Loss layer ###

class cross_entropy :

    def __init__(self, y_pred, y_true) :

        self.y_pred = np.clip(y_pred, 10e-7, 1 - 10e-7)
        self.y_true = y_true

    def compute(self) :

        error = np.dot(np.transpose(np.log(self.y_pred)),self.y_true)

        return -np.mean(error)

    def grad(self) :

        grad = -np.sum(np.divide(self.y_true,self.y_pred))

        return grad/grad.size

### Normalize layer ###

class normalize_layer(layer) :

    def __init__(self) :
        pass

    def forward(self,input) :
        pass

    def backward(self,grad,eta) :
        pass
















