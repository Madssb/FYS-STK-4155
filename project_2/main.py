import numpy as np
import jax.numpy as jnp
from jax import grad
from ffnn import FeedForwardNeuralNetwork
from gradient_descents import GradientDescent
from activation_functions import sigmoid
from loss_functions import mean_squared_error



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
  output = forward_pass_func(parameters, features)
  return loss_func(target, output)


def grad_meta_loss(parameters: tuple, target, features):
  """Compute gradient of meta loss w.r.t parameters
  """
  return jnp.array(grad(meta_loss)(parameters, target, features))


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
  instance = GradientDescent(learning_rate, grad_meta_loss, parameters, y_xor, features)
  trained_parameters, max_iter_met, converged = instance(max_iter, tolerance)
  network.insert_parameters(trained_parameters)
  output = network(features)
  print(output)

if __name__ == "__main__":
    simple_ffnn_xor()