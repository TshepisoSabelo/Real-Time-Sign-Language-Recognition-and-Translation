import numpy as np
from activation import ReLu, Softmax

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
class Convolutional(Layer):
    """Apply valid 2-D cross-correlation across multi-channel inputs."""

    def __init__(self, input_shape, kernel_size, layer_size, activation = ReLu):
        """Initialize a convolutional layer for the supplied tensor dimensions.

        Args:
            input_shape: Input dimensions as ``(channels, height, width)``.
            kernel_size: Height and width of each square kernel.
            layer_size: Number of kernels in the layer.
        """
        super().__init__()
        self.beta = 0.85
        self.t = 0
        self.input_shape = input_shape
        self.depth = layer_size
        self.input_channels, self.input_height, self.input_width = input_shape #(3, 28, 28)

        self.output_shape = (self.depth, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1) #for a valid conv with no padding
        self.layer_shape = (self.depth, self.input_channels, kernel_size, kernel_size) #(5, 3, 3, 3)

        fan_in = self.layer_shape[1] * self.layer_shape[2] * self.layer_shape[3]
        self.kernels = (np.random.randn(*self.layer_shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.biases = np.zeros((self.depth, 1, 1), np.float32) #(1 bias per 3 channels-kernel)

        self.kernels_velocity = np.zeros(self.layer_shape, dtype=np.float32)
        self.biases_velocity = np.zeros((self.depth, 1, 1), dtype=np.float32)
        self.kernels_gradient = np.zeros_like(self.kernels)
        self.biases_gradient = np.zeros_like(self.biases)

        self.activation = activation()

    def forward(self, input):
        """Convolve ``input`` with vectorized sliding windows."""
        self.input = np.asarray(input, dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(
            self.input, (self.layer_shape[2], self.layer_shape[3]), axis=(1, 2)
        )
        self.output = np.einsum("cxyij,ocij->oxy", windows, self.kernels, optimize=True)
        self.output += self.biases
        return self.activation.forward(self.output)
    
    def zero_grad(self):
        self.kernels_gradient.fill(0.0)
        self.biases_gradient.fill(0.0)

    def backward(self, output_gradient, learning_rate=None):
        """Accumulate gradients and optionally update for compatibility."""
        grad = np.asarray(output_gradient, dtype=np.float32) * self.activation.backward(self.output)
        windows = np.lib.stride_tricks.sliding_window_view(
            self.input, (self.layer_shape[2], self.layer_shape[3]), axis=(1, 2)
        )
        kernels_gradient = np.einsum("cxyij,oxy->ocij", windows, grad, optimize=True)
        input_gradient = np.zeros_like(self.input, dtype=np.float32)

        for kernel_row in range(self.layer_shape[2]):
            for kernel_col in range(self.layer_shape[3]):
                input_gradient[:, kernel_row:kernel_row + self.output_shape[1], kernel_col:kernel_col + self.output_shape[2]] += np.einsum(
                    "oxy,oc->cxy", grad, self.kernels[:, :, kernel_row, kernel_col], optimize=True
                )

        self.kernels_gradient += kernels_gradient
        self.biases_gradient += np.sum(grad, axis=(1, 2), keepdims=True, dtype=np.float32)
        if learning_rate is not None:
            self.update_parameters(learning_rate, 1)
        return input_gradient

    def update_parameters(self, learning_rate, batch_size):
        scale = np.float32(1.0 / batch_size)
        self.kernels_velocity = self.beta * self.kernels_velocity + (1 - self.beta) * self.kernels_gradient * scale
        self.biases_velocity = self.beta * self.biases_velocity + (1 - self.beta) * self.biases_gradient * scale
        self.t += 1
        correction = np.float32(1.0 - self.beta ** self.t)
        self.kernels -= np.float32(learning_rate) * self.kernels_velocity / correction
        self.biases -= np.float32(learning_rate) * self.biases_velocity / correction
        self.zero_grad()

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
        grad = np.reshape(output_gradient, self.input_shape)
        #print(f"input shape: {self.input_shape}")
        #print(f"flatten layer output grad shape: {grad.shape}")
        return grad

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

    def __init__(self, input_shape, output_size, activation):
        """Initialize the layer's weights and biases.
            
        Args:
            input_shape: The shape of the data recieved from the previous layer
            output_shape: The expected output shape
            activation: Activation function to use at the end of the layer
            y_true: For cases when Softmax is used for activation.
        """
        super().__init__()
        self.beta = 0.8 #for momentum gradient descent optimization
        self.input_shape = input_shape
        self.output_size = output_size
        self.weights_velocity = np.zeros((output_size, input_shape[0]), np.float32)
        self.weights = (np.random.randn(output_size, input_shape[0]) * np.sqrt(2.0 / input_shape[0])).astype(np.float32)

        self.biases_velocity = np.zeros((output_size, 1), np.float32)
        self.biases = np.zeros((output_size, 1), np.float32)
        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

        self.activation = activation()

    def forward(self, input):
        """Compute the linear transformation of ``input``."""
        #print(f"Dense layer input shape: {input.shape}")
        self.input = np.asarray(input, dtype=np.float32)
        self.output = np.dot(self.weights, self.input) + self.biases
        #print(f"Dense layer output: {self.output}")
        return self.activation.forward(self.output)

    def zero_grad(self):
        self.weights_gradient.fill(0.0)
        self.biases_gradient.fill(0.0)

    def backward(self, output_gradient, y_true=None, learning_rate=None):
        """Accumulate gradients and optionally update for compatibility."""
        if isinstance(self.activation, Softmax):
            output_gradient = self.activation.backward(self.output, y_true)
        else:
            output_gradient = np.asarray(output_gradient, dtype=np.float32) * self.activation.backward(self.output)

        output_gradient = np.asarray(output_gradient, dtype=np.float32).reshape(-1, 1)
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights_gradient += weights_gradient
        self.biases_gradient += output_gradient
        if learning_rate is not None:
            self.update_parameters(learning_rate, 1)
        return input_gradient

    def update_parameters(self, learning_rate, batch_size):
        scale = np.float32(1.0 / batch_size)
        self.weights_velocity = self.beta * self.weights_velocity + (1 - self.beta) * self.weights_gradient * scale
        self.biases_velocity = self.beta * self.biases_velocity + (1 - self.beta) * self.biases_gradient * scale
        self.weights -= np.float32(learning_rate) * self.weights_velocity
        self.biases -= np.float32(learning_rate) * self.biases_velocity
        self.zero_grad()

class Dropout(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

    def backward(self, gradient, learning_rate=None):
        return gradient


# ---- Test script ----
def check_conv_weights_update():
    np.random.seed(42)

    # Small input: 3 channels, 8x8 image
    input_shape = (3, 8, 8)
    kernel_size = 3
    layer_size = 5
    learning_rate = 0.05

    layer = Convolutional(input_shape, kernel_size, layer_size)

    # Snapshot weights before
    kernels_before = layer.kernels.copy()
    biases_before = layer.biases.copy()

    # Random input
    x = np.random.randn(*input_shape)

    # Forward
    y = layer.forward(x)
    print(f"Forward output shape: {y.shape} (expected {layer.output_shape})")
    assert y.shape == layer.output_shape, "Output shape mismatch!"

    # Fake upstream gradient (same shape as output)
    output_gradient = np.random.randn(*layer.output_shape)

    # Backward
    input_gradient = layer.backward(output_gradient, learning_rate)

    # Snapshot weights after
    kernels_after = layer.kernels.copy()
    biases_after = layer.biases.copy()

    # ---- Checks ----
    print("\n=== Weight update checks ===")
    print(f"Kernels changed:  {not np.allclose(kernels_before, kernels_after)}")
    print(f"Biases changed:   {not np.allclose(biases_before, biases_after)}")

    print(f"Max |Δkernel|:    {np.max(np.abs(kernels_after - kernels_before)):.6e}")
    print(f"Max |Δbias|:      {np.max(np.abs(biases_after - biases_before)):.6e}")

    # Sanity: input gradient shape
    print(f"\nInput gradient shape: {input_gradient.shape} (expected {input_shape})")
    assert input_gradient.shape == input_shape, "Input gradient shape mismatch!"

    # ---- Directional check: update should reduce loss ----
    print("\n=== Directional check (loss should decrease) ===")

    # Use a simple MSE-like loss w.r.t. a random target
    target = np.random.randn(*layer.output_shape)

    def loss_of(layer_obj, x_in, tgt):
        out = layer_obj.forward(x_in)
        return 0.5 * np.sum((out - tgt) ** 2)

    layer2 = Convolutional(input_shape, kernel_size, layer_size)
    layer2.kernels = kernels_before.copy()
    layer2.biases = biases_before.copy()

    loss_before = loss_of(layer2, x, target)

    # Gradient of MSE w.r.t. output = (out - target)
    out_before = layer2.forward(x)
    dloss = (out_before - target)

    layer2.backward(dloss, learning_rate)

    loss_after = loss_of(layer2, x, target)

    print(f"Loss before step: {loss_before:.6f}")
    print(f"Loss after step:  {loss_after:.6f}")
    print(f"Loss decreased:   {loss_after < loss_before}")

    # ---- Multiple-step check ----
    print("\n=== Multi-step training check ===")
    layer3 = Convolutional(input_shape, kernel_size, layer_size)
    init_norm = np.linalg.norm(layer3.kernels)

    losses = []
    for step in range(200):
        out = layer3.forward(x)
        dloss = (out - target)
        layer3.backward(dloss, learning_rate)
        losses.append(0.5 * np.sum((out - target) ** 2))

    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss:   {losses[-1]:.6f}")
    print(f"Kernel norm changed: {init_norm:.6f} -> {np.linalg.norm(layer3.kernels):.6f}")
    print(f"Loss decreased over training: {losses[-1] < losses[0]}")



