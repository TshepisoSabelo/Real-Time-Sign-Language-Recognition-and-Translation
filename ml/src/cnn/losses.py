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

def categorical_cross_entropy(y_true: list[float], y_pred: list[float]):
    loss = 0
    for i in range(len(y_true)):
        loss -= (math.log(y_pred[i]) + y_true [i])
        
    return loss

def categorical_cross_entropy_prime(y_true: list[float], y_pred: list[float]):
    derivative = []
    for i in range(len(y_true)):
        derivative.append(-y_true[i] / y_pred[i])

    return derivative