import numpy as np

def sigmoid(x):
    """Compute out for some neuron.

    Parameters
    ----------
    activation: np.ndarray
      Activation.
    
    Returns
    -------
      np.ndarray
    """
    return 1./(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return np.heaviside(x, np.zeros_like(x))

def leaky_ReLU(x):
    return np.maximum(0.01*x, x)

def leaky_ReLU_derivative(x, alpha=0.01):
    return np.where(x>0, 1, alpha)

def identity(x):
    return x