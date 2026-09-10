import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
class Layer ():
    """Abstract base class for neural-network layers."""

    def __init__(self):
        """Initialize storage for a layer's input and output tensors."""
        self.input = None
        self.output = None

    def forward(self, input):
        """Compute and return the layer output for ``input``."""
        pass

    def backward(self, output_gradient, learning_rate=None):
        """Propagate ``gradient`` backward and return the input gradient."""
        pass

    def updateParameters(self):
        """Update trainable parameters using accumulated gradients."""
        pass

#-----------------------------------------------------------------------------------------------------------------------
from scipy import signal
class Convolutional(Layer):
    """Apply valid 2-D cross-correlation across multi-channel inputs."""

    def __init__(self, input_shape, kernel_size, layer_size):
        """Initialize a convolutional layer for the supplied tensor dimensions.

        Args:
            input_shape: Input dimensions as ``(channels, height, width)``.
            kernel_size: Height and width of each square kernel.
            layer_size: Number of kernels in the layer.
        """
        super().__init__()

        input_channels, input_height, input_width = input_shape
        self.depth = layer_size
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.output_shape = (self.depth, input_height - kernel_size + 1, input_width - kernel_size + 1) #for a valid conv with no padding
        self.kernels_shape = (self.depth, input_channels, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = self.biases = np.random.randn(self.depth, 1, 1)

    def forward(self, input):
        """Convolve ``input`` with the layer kernels and return the output."""
        self.input = input
        self.output = np.zeros(self.output_shape)

        for i in range(self.depth):
            for j in range(self.input_channels):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

        self.output += self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """Propagate gradients and update kernels and biases in place."""
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i  in range(self.depth):
            for j in range(self.input_channels):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i,j], "full")

        bias_gradient = np.sum(output_gradient, axis=(1, 2), keepdims=True)
        self.biases -= learning_rate * bias_gradient
        return input_gradient

#-----------------------------------------------------------------------------------------------------------------------
class Reshape(Layer):
    """Reshape a tensor during the forward and backward passes."""

    def __init__ (self,input_shape, output_shape):
        """Initialize the input and output tensor shapes."""
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        """Return ``input`` reshaped to the configured output shape."""
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient):
        """Reshape the output gradient back to the input shape."""
        return np.reshape(output_gradient, self.input_shape)

#-----------------------------------------------------------------------------------------------------------------------
class Flatten(Layer):
    """Flatten an input tensor into a column vector."""

    def forward(self, input):
        """Return the input values as a column vector."""
        self.input_shape = input.shape
        return np.reshape(input, (-1, 1))

    def backward(self, output_gradient, learning_rate=None):
        """Reshape the output gradient back to the input tensor shape."""
        return np.reshape(output_gradient, self.input_shape)

#-----------------------------------------------------------------------------------------------------------------------

#Pooling Layers
class MaxPoolingLayer(Layer):

    def __init__(self):
        """Initialize the max-pooling layer."""
        super().__init__()

    def forward(self, input):
        """Forward ``input`` through the base layer implementation."""
        return super().forward(input)

    def backward(self, gradient):
        """Backward-propagate ``gradient`` through the base implementation."""
        return super().backward(gradient)

#-----------------------------------------------------------------------------------------------------------------------
class Dense(Layer):
    """Fully connected layer that applies a linear transformation."""

    def __init__(self, input_shape, output_size):
        """Initialize the layer's weights and biases."""
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_shape[0])
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input):
        """Compute the linear transformation of ``input``."""
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        """Propagate gradients and update weights and biases in place."""
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Update weights and biases
        self.weights += learning_rate * weights_gradient
        self.biases += learning_rate * output_gradient

        return input_gradient

# print("DENSE LAYER TEST: BASIC SHAPE VERIFICATION")
# print("=" * 70)

# dense = Dense((64,1), 29)

# # Create random input
# input_data = np.random.randn(64, 1)
# print(f"\nInput shape: {input_data.shape}")

# # Forward pass
# output = dense.forward(input_data)
# print (output[0])
# print(f"Output shape: {output.shape}")  # (29, 1)

# # Simulate gradient for backward pass
# output_gradient = np.random.randn(29, 1)
# print(f"Output gradient shape: {output_gradient.shape}")

# # Backward pass
# input_gradient = dense.backward(output_gradient, learning_rate=0.01)

# print(f"\nGradient shapes:")
# print(f"  weights_gradient: {dense.weights.shape}")  # (29, 64)
# print(f"  biases_gradient: {dense.biases.shape}")    # (29, 1)
# print(f"  input_gradient: {input_gradient.shape}")   # (64, 1)

# # Verify shapes match
# assert dense.weights.shape == (29, 64), "Weights shape wrong!"
# assert dense.biases.shape == (29, 1), "Biases shape wrong!"
# assert input_gradient.shape == (64, 1), "Input gradient shape wrong!"


