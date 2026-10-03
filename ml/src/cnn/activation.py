import   numpy as np
class Activation():
     def __init__(self) -> None:
          pass
     
     def forward(self, x):
          pass
     
     def backward(self, grad):
          pass
#===============================================================================================
class Tanh(Activation):
    def __init__(self) -> None:
          super.__init__
    
    def foward(self, x):
        return np.tanh(x)

    def backward(self, grad):
        return 1 - np.tanh(grad) ** 2

class Sigmoid(Activation):
    def __init__(self) -> None:
          super().__init__()

    def foward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, grad):
        s = self.foward(grad)
        return s * (1 - s)

class ReLu(Activation):
    def __init__(self) -> None:
          super().__init__()

    def forward(self, x):
        return np.maximum(np.float32(0.0), np.asarray(x, dtype=np.float32)).astype(np.float32, copy=False)

    def backward(self, grad):
        return (np.asarray(grad) > 0).astype(np.float32)

class Softmax(Activation):
    def __init__(self) -> None:
          super().__init__()

    def forward(self, input):
        # subtract max before exp to bring the numbers to some sort of normalized range, to avoid overflow
        #input = input.flatten()  # Flatten the input to a 1D array
        input = np.asarray(input, dtype=np.float32)
        input_shifted = input - np.max(input)
        tmp = np.exp(input_shifted, dtype=np.float32)
        output = tmp / np.sum(tmp, dtype=np.float32)
        return output.reshape(-1, 1).astype(np.float32, copy=False)
    
    # def Softmax_prime(output, output_gradient):
    #     # Convert to column vectors
    #     p = softmax(output).reshape(-1, 1)  # (n_classes, 1)
    #     grad = output_gradient.reshape(-1, 1)  # (n_classes, 1)
        
    #     # Softmax Jacobian: J = diag(p) - p·pᵀ
    #     # This is an (n_classes x n_classes) matrix
    #     jacobian = np.diagflat(p) - np.dot(p, p.T)
        
    #     # Apply chain rule: dL/dz = J · dL/dp
    #     grad_input = jacobian @ grad
        
    #     # Return as 1D array
    #     return grad_input

    def backward(self, logits, y_true):
        """
        Gradient of softmax + cross-entropy w.r.t. logits z.
        (Assuming y_pred = softmax(z))
        
        Returns:
            grad: shape (n_classes, 1)
        """
        return (np.asarray(logits, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)).reshape(-1, 1)

# softmax = Softmax()
# logits = np.array([1000, 2000, 3000, 1500, 2500])
# error = np.array([0.0, 0.0, 1.0, 0.0, 0.0]) - softmax.forward(logits)