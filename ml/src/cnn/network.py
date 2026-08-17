import losses
from layers import Layer
class Network():
    def __init__(self) -> None:
        self.layers = []

    def add_layer(self, layer:Layer):
        self.layers.append(layer)

    def predict(self, network, input):
        output = input
        for layer in network:
            output = layer.forward()
        return output
    
    def train(self, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
        for e in range(epochs):
            error = 0
            for x, y in (x_train, y_train):
                #forward pass
                output = self.predict(self.layers, x)
                #error
                error += loss(y, output)

                #backward propagatation
                grad = loss_prime(y, output)
                for layer in self.layers:
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            if verbose:
                print(f"{e + 1}/{epochs}, error = {error}")

