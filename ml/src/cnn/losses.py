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
    """
    Compute categorical cross-entropy loss.
    
    Args:
        y_true: One-hot encoded true labels, shape (n_classes,)
        y_pred: Predicted probabilities, shape (n_classes,)
                Should sum to 1 (after softmax)
    
    Returns:
        loss: Scalar loss value
    """
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Cross-entropy loss: -Σ y_true * log(y_pred)
    loss = -np.sum(y_true * np.log(y_pred))
        
    return loss

def categorical_cross_entropy_prime(y_true: list[float], y_pred: list[float]):
    derivative = []
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    for i in range(len(y_true)):
        derivative.append(-y_true[i] / y_pred[i])

    grad = np.array(derivative)
    grad = grad.reshape(-1, 1)
    return grad

y_true = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y_pred = [0, 0.23, 0.87, 0.03, 0.024, 0.01, 0.005, 0.001, 0.0001, 0.00001]
derivative = categorical_cross_entropy(y_true, y_pred)
print(derivative)