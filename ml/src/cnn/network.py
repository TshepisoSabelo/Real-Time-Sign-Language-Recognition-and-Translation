from layers import Layer, Dense
from time import perf_counter
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
            #print(f"\nOutput Shape: {output.shape}")
        return output
    
    def validation(self, X_val, y_val):
        return self._evaluate(X_val, y_val, loss=None)[1]

    def _evaluate(self, X_val, y_val, loss=None):
        correct_pred = 0
        total = 0
        total_loss = 0.0
        if hasattr(X_val, "iter_batches"):
            batches = X_val.iter_batches(batch_size=32, shuffle=False)
            batch_values = ((x, y) for x_batch, y_batch, _ in batches for x, y in zip(x_batch, y_batch))
        else:
            batch_values = zip(X_val, y_val)

        for x, y in batch_values:

            y = y.reshape(-1, 1)
            y_pred = self.predict(self.layers, x)
            if loss is not None:
                total_loss += float(loss(y, y_pred))

            if np.argmax(y_pred) == np.argmax(y):
                correct_pred += 1
            total += 1

        average_loss = total_loss / total if total and loss is not None else 0.0
        accuracy = correct_pred / total if total else 0.0
        return average_loss, accuracy
    
    def _batch_iterator(self, x_train, y_train, batch_size, shuffle):
        if hasattr(x_train, "iter_batches"):
            yield from x_train.iter_batches(batch_size=batch_size, shuffle=shuffle)
            return

        indices = np.arange(len(x_train))
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            yield (
                np.asarray(x_train[selected], dtype=np.float32),
                np.asarray(y_train[selected], dtype=np.float32),
                0.0,
            )

    def train(
        self,
        x_train,
        y_train=None,
        X_val=None,
        y_val=None,
        loss=None,
        epochs=10,
        learning_rate=0.01,
        batch_size=32,
        verbose=True,
        shuffle=True,
        profile=False,
    ):
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

        history = {
            "batch_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch": [],
        }

        for e in range(epochs):
            correct_preds = 0
            samples_seen = 0
            epoch_start = perf_counter()
            loading_seconds = 0.0
            forward_seconds = 0.0
            backward_seconds = 0.0
            update_seconds = 0.0
            batch_count = 0

            for x_batch, y_batch, load_seconds in self._batch_iterator(x_train, y_train, batch_size, shuffle):
                batch_count += 1
                loading_seconds += load_seconds
                actual_batch_size = len(x_batch)
                for layer in self.layers:
                    if hasattr(layer, "zero_grad"):
                        layer.zero_grad()

                batch_loss = 0.0
                batch_start = perf_counter()
                batch_backward_seconds = 0.0
                for x, y in zip(x_batch, y_batch):
                    y = y.reshape(-1, 1).astype(np.float32, copy=False)
                    y_pred = self.predict(self.layers, x)
                    batch_loss += loss(y, y_pred) if loss is not None else 0.0

                    if np.argmax(y_pred) == np.argmax(y):
                        correct_preds += 1

                    samples_seen += 1
                    grad = y
                    backward_start = perf_counter()

                    #start backward propagatioon
                    for layer in reversed(self.layers):
                        if isinstance(layer, Dense):
                            grad = layer.backward(grad, y, learning_rate=None)
                        else:
                            grad = layer.backward(grad, learning_rate=None)
                    batch_backward_seconds += perf_counter() - backward_start

                backward_seconds += batch_backward_seconds
                forward_seconds += perf_counter() - batch_start - batch_backward_seconds
                update_start = perf_counter()

                #only update parameters noe after each batch processing
                for layer in reversed(self.layers):
                    if hasattr(layer, "update_parameters"):
                        layer.update_parameters(learning_rate, actual_batch_size)
                update_seconds += perf_counter() - update_start
                history["batch_loss"].append(batch_loss / actual_batch_size if loss is not None else 0.0)

                if profile and verbose:
                    print(
                        f"\rBatch {batch_count}: loss={batch_loss / actual_batch_size:.4f}",
                        end="",
                        flush=True,
                    )

            train_accuracy = correct_preds / samples_seen if samples_seen else 0.0
            has_lazy_validation_labels = hasattr(X_val, "iter_batches")
            if X_val is not None and (y_val is not None or has_lazy_validation_labels):
                val_loss, val_accuracy = self._evaluate(X_val, y_val, loss=loss)
            else:
                val_loss, val_accuracy = 0.0, 0.0
            history["epoch"].append(e + 1)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            epoch_seconds = perf_counter() - epoch_start
            if verbose:
                if profile:
                    print()
                print(f"Epoch {e + 1}/{epochs}: train_accuracy={train_accuracy:.3f}, val_accuracy={val_accuracy:.3f}")
            if profile:
                average_batch = epoch_seconds / batch_count if batch_count else 0.0
                print(
                    f"Timing: load={loading_seconds:.3f}s, forward={forward_seconds:.3f}s, "
                    f"backward={backward_seconds:.3f}s, update={update_seconds:.3f}s, "
                    f"epoch={epoch_seconds:.3f}s, batch={average_batch:.3f}s"
                )

        return history
        

    def test(self, x_train, y_train, loss, loss_prime, activation):
        error = 0
        b = 0

        #forward pass
        output = self.predict(self.layers, x_train)
        y = y_train.reshape(-1, 1)
        #print(y.shape)
        #print(output[0])
        #error
        output = activation(output)
        error += loss(y, output)
        print(error)

        #back propagation
        output_gradient = loss_prime(y_train, output)
        print(f"\nActivation output gradient shape: {output_gradient.shape}")

        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                output_gradient = layer.backward(output_gradient, y, 0.02)
            else:
                output_gradient = layer.backward(output_gradient, 0.02)
            print(f"\nOutput_grad shape: {output_gradient.shape}")


# model = Network()

# model.add_layer(Convolutional(input_shape=(3, 28, 28), kernel_size=3, layer_size=8))
# model.add_layer(Flatten())
# model.add_layer(Dense(input_shape=(5408, 1), output_size=29))
# #followed by softmax activation
# #then the loss function

# input_data = np.random.randn(3, 28, 28)
# Y = np.random.randn(29, 1)
# print(f"\nInput shape: {input_data.shape}")
# #model.train(loss = losses.categorical_cross_entropy, loss_prime = losses.categorical_cross_entropy_prime, activation = softmax, activation_prime = softmax_prime, x_train = x_train, y_train = y_train, epochs = 2, learning_rate = 0.01, verbose = True)
# model.test(x_train= input_data, y_train=Y, activation = softmax, activation_prime = softmax_prime, loss=losses.categorical_cross_entropy, loss_prime = losses.categorical_cross_entropy_prime)


