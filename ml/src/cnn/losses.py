import numpy as np
import math 
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Cross-entropy loss: -Σ y_true * log(y_pred)
    loss = -np.sum(y_true * np.log(y_pred))

    return loss

# y_true = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# y_pred = [0.9214, 0.0021, 0.0338, 0.0011, 0.0003, 0.0122, 0.0004, 0.0281, 0.0004, 0.0002]
# loss = Softmax_cross_entropy_prime(y_true, y_pred)
# print(loss)