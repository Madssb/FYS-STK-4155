import numpy as np
import jax.numpy as jnp
from jax import grad
from flatten_n_unflatten import flatten
from gradient_descents import GradientDescent


def activation_function(activation):
    return 1 / (1 + jnp.exp(-activation))


features = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
features = features.T
#_number denotes which layer 
weights_1 = np.ones((2, 2), dtype=float)
biases_1 = np.random.randn(2, 1)
weights_2 = np.ones((2, 1))
bias_2 = np.random.normal()


parameters = (weights_1, weights_2, biases_1, bias_2)
parameters_tuple, reshape_func = flatten(parameters)



def simple_ffnn(parameters_tuple, features):
  """
  Feed forwark neural network
  """
  weights_1, weights_2, biases_1, bias_2 = reshape_func(parameters_tuple)
  activation_1 = weights_1.T @ features + biases_1
  activation_2 = weights_2.T @ activation_function(activation_1) + bias_2
  target_output = activation_function(activation_2) 
  return target_output




a = simple_ffnn(parameters_tuple, features)


def loss(target, output):
  """
  Compute mean squared error.
  """
  return jnp.mean((target - output)**2)



def meta_loss(parameters_tuple, target, features):
  """
  Compute MSE for FFNN
  """
  model_output = simple_ffnn(parameters_tuple, features)
  mse = loss(target, model_output)
  return mse


def grad_meta_loss(parameters_tuple, target, features):
  """
  Compute gradient of meta loss w.r.t parameters
  """
  return jnp.array(grad(meta_loss)(parameters_tuple, target, features))


y_xor = jnp.array([0,1,1,1])
learning_rate = 0.1
max_iter = 2000
tolerance = 1e-5

instance = GradientDescent(learning_rate, grad_meta_loss, parameters_tuple, y_xor, features)
optimized_parameters, max_iter_met, converged = instance(max_iter, tolerance)
model_output = simple_ffnn(optimized_parameters, features)

print(f"{y_xor=}")
print(f"{model_output=}")
print(f"{max_iter_met=}")
print(f"{converged=}")