import random
import numpy as np

class Kernel():
    def __init__(self, shape):
        self.depth, self.size = shape
        self.weights = []
        self.generate_weights()

    def generate_weights(self):
        rng = np.random.default_rng()

        for i in range(self.depth):
            matrix = rng.random((self.size, self.size))
            self.weights.append[matrix]

    def get_kernel(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

from abc import ABC, abstractmethod
class Layer (ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, gradient):
        pass

    @abstractmethod
    def updateParameters(self):
        pass

import numpy as np
import random
from scipy import signal
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, layer_size, depth):
        super().__init__()

        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.layer_size = layer_size
        self.layer_shape = (depth, layer_size)
        self.bias_size = layer_size
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid", "fill")
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i  in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i,j], "full")

        self.kernels -= learning_rate*kernels_gradient
        self.biases -= learning_rate*output_gradient
        return input_gradient

#Pooling Layers
    class MaxPoolingLayer(Layer):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return super().forward(input)

        def backward(self, gradient):
            return super().backward(gradient)