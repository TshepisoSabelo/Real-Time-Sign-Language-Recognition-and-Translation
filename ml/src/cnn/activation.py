import   numpy as np
    
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.max(0, x)
def relu_prime(x):
    return np.where(x > 0, 1, 0)

def softmax(input):
        # subtract max before exp to bring the numbers to some sort of normalized range, to avoid overflow
        #input = input.flatten()  # Flatten the input to a 1D array
        input_shifted = input - np.max(input)
        tmp = np.exp(input_shifted)
        output = tmp / np.sum(tmp)
        return output
    
def softmax_prime(output, output_gradient):
        # Convert to column vectors
        p = softmax(output).reshape(-1, 1)  # (n_classes, 1)
        grad = output_gradient.reshape(-1, 1)  # (n_classes, 1)
        
        # Softmax Jacobian: J = diag(p) - p·pᵀ
        # This is an (n_classes x n_classes) matrix
        jacobian = np.diagflat(p) - np.dot(p, p.T)
        
        # Apply chain rule: ∂L/∂z = J · ∂L/∂p
        grad_input = jacobian @ grad
        
        # Return as 1D array
        return grad_input.flatten()

#softmax = Softmax()
#ogits = np.array([1000, 2000, 3000, 1500, 2500])
#error = np.array([0.0, 0.0, 1.0, 0.0, 0.0]) - softmax.forward(ogits)
#print(error)