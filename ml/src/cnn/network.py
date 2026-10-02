from layers import Dense, Layer
from layers import Convolutional, Flatten
from data_loader import load_dataset
import losses
import time
from activation import softmax, softmax_prime, tanh, sigmoid, relu
import numpy as np

DEFAULT_INPUT_SHAPE = (3, 28, 28)  # (channels, height, width)

class Network():
    """Container for an ordered sequence of neural-network layers."""

    def __init__(self) -> None:
        """Initialize an empty network."""
        self.layers = []

    def add_layer(self, layer:Layer):
        """Append ``layer`` to the network's execution sequence."""
        self.layers.append(layer)

    def predict(self, network, input):
        """Run ``input`` through each layer in ``network``."""
        output = input
        for layer in network:
            output = layer.forward(output)
        return output
    
    def train(self, loss, loss_prime, activation, activation_prime, x_train, y_train, epochs = 2, learning_rate = 0.002, verbose = True, batch_size = 32):
        """Train the network using forward propagation and gradient descent.

        Args:
            loss: Function that calculates the prediction error.
            loss_prime: Function that calculates the loss gradient.
            x_train: Training inputs.
            y_train: Expected outputs corresponding to ``x_train``.
            epochs: Number of complete passes through the training data.
            learning_rate: Step size used when updating layer parameters.
            verbose: Whether to print the error after each epoch.
        """
        for e in range(epochs):
            print (f"Epoch: {e}")

            #for b in range(0, len(x_train), batch_size):
            #  print(f"Training Batch: {b/32}")
            start = time.time()
                #x_batch = x_train[b:b + batch_size]
                #y_batch = y_train[b:b + batch_size]

            for x, y in zip(x_train, y_train):
                error = 0
                #forward pass
                output = self.predict(self.layers, x)
                output = activation(output)
                error += loss(y, output)

                print("Output NaN:", np.isnan(output).any())
                print("Output max:", np.max(output))
                print("Output min:", np.min(output))

                print("Error:", error)
                grad = loss_prime(y, output)
                print("Gradient NaN:", np.isnan(grad).any())
                #backward propagatation

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            end = time.time()
            
            if verbose:
                print(f"{e + 1}/{epochs}, error = {error}")

    def test(self, x_train, y_train, loss, activation, batch_size = 32):
        error = 0
        b = 0
        x_batch = x_train[b:b + batch_size]
        y_batch = y_train[b:b + batch_size]
        for x, y in zip(x_batch, y_batch):
            #forward pass
            output = self.predict(self.layers, x)
            print(f"conv_layer output shape: {output.shape}")
            print("output:")
            print(output)
            y = y.reshape(-1, 1)
            #print(y.shape)
            #print(output[0])
            #error
            output = activation(output)
            error += loss(y, output)
            print(error)


model = Network()

# model.add_layer(Convolutional(input_shape=(3, 28, 28), kernel_size=3, layer_size=8))
# #model.add_layer(Convolutional(input_shape=(8, 26, 26), kernel_size=3, layer_size=8))
# #model.add_layer(Convolutional(input_shape=(8, 24, 24), kernel_size=3, layer_size=8))
# # model.add_layer(Flatten())
# # #flattened output shape will be (8 * 26 * 26,) = (5408,1)
# # model.add_layer(Dense(input_shape=(5408, 1), output_size=128))
# # model.add_layer(Dense(input_shape=(128, 1), output_size=64))
# # model.add_layer(Dense(input_shape=(64, 1), output_size=29))
# #followed by softmax activation
# #then the loss function

# #debugger
# x_train, y_train = load_dataset()
# print(x_train.shape)
# print(y_train.shape)
# print(y_train[0])
# print(f"Number of samples: {len(x_train)}")

    
# #model.train(loss = losses.categorical_cross_entropy, loss_prime = losses.categorical_cross_entropy_prime, activation = softmax, activation_prime = softmax_prime, x_train = x_train, y_train = y_train, epochs = 2, learning_rate = 0.01, verbose = True)
# model.test(x_train=x_train, y_train=y_train, activation = softmax, loss=losses.categorical_cross_entropy)


