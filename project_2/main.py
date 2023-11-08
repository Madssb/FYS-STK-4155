import numpy as np
import jax.numpy as jnp
from jax import grad
from ffnn import FeedForwardNeuralNetwork
from gradient_descents import GradientDescent
from activation_functions import sigmoid
from loss_functions import mean_squared_error


import numpy as np

def meta_loss(parameters: tuple, forward_pass_func: callable, loss_func: callable,
              features: np.ndarray, target: np.ndarray):
    """Compute loss of model w.r.t target.

    Parameters
    ----------
    parameters : tuple of float
        Parameters for FFNN.
    forward_pass_func : callable
        Callable representing the forward pass of the neural network.
        It is expected to be a callable representing the `meta_forward_pass` method of 
        the `FeedForwardNeuralNetwork` class.
    loss_func : {mean_squared_error, cross_entropy}
        Loss function.
    features : np.ndarray
        Features.
    target : np.ndarray
        Target.
    """
    if not isinstance(parameters, tuple):
        raise TypeError(f"parameters: expected tuple, got {type(parameters)}.")
    # for param in parameters:
    #     if not isinstance(param, float):
    #         raise TypeError(f"Each element in parameters should be a float, but found {type(param)}.")
    if not callable(forward_pass_func):
        raise TypeError(f"forward_pass_func: expected callable, got {type(forward_pass_func)}.")
    if not callable(loss_func):
        raise TypeError(f"loss_func: expected callable, got {type(loss_func)}.")
    if not isinstance(features, (np.ndarray, jnp.ndarray)):
        raise TypeError(f"features: expected np.ndarray, got {type(features)}.")
    if not isinstance(target,( np.ndarray, jnp.ndarray)):
        raise TypeError(f"target: expected np/jnp.ndarray, got {type(target)}.")

    output = forward_pass_func(parameters, features)
    return loss_func(target, output)



def grad_meta_loss(parameters: tuple, meta_loss_func: callable):
    """
    compute grad of meta_loss w.r.t parameters
    """
    if not isinstance(parameters, tuple):
        raise TypeError(f"parameters: expected tuple, got {type(parameters)}.")
    # for param in parameters:
    #     if not isinstance(param, float):
    #         raise TypeError(f"Each element in parameters should be a float, but found {type(param)}.")
    grad_func = grad(meta_loss_func)
    return jnp.array(grad_func(parameters))



def simple_ffnn_xor():
  y_xor = jnp.array([0,1,1,1])
  features = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
  features = features.T
  network = FeedForwardNeuralNetwork(2,1,2,1, sigmoid)
  network.initialize_weights_and_biases()
  parameters = network.extract_sparameters()
  learning_rate = 0.1
  tolerance = 1e-5
  max_iter = 1000
  meta_mse = lambda params: meta_loss(params, network.meta_forward_pass, 
                                      mean_squared_error, features, y_xor)
  grad_meta_mse = lambda params: grad_meta_loss(params, meta_mse)
  optimizer = GradientDescent(learning_rate, grad_meta_mse, parameters)
  trained_parameters, max_iter_met, converged = optimizer(max_iter, tolerance)
  network.insert_parameters(trained_parameters)
  output = network(features)
  print(output)

if __name__ == "__main__":
    simple_ffnn_xor()