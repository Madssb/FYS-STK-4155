import numpy as np
import jax.numpy as jnp
from jax import grad
import warnings
from ffnn import FeedForwardNeuralNetwork
from gradient_descents import *
from activation_functions import sigmoid
from loss_functions import mean_squared_error

import matplotlib.pyplot as plt

def poly_features(input, degree):
    """
    Create a design matrix for polynomial regression.

    Parameters:
    - input (numpy.ndarray): Input data points.
    - degree (int): Degree of the polynomial.

    Returns:
    - design_matrix (numpy.ndarray): Design matrix with polynomial features.
    """
    n = len(input)
    design_matrix = np.zeros((n, degree + 1))

    for i in range(degree + 1):
        design_matrix[:, i] = input ** i

    return design_matrix

def cost_grad_func(features, target, parameters):
    """
    cost function differentiated w.r.t parameters
    """
    return -2*features.T @ (target - features @ parameters)

def franke_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """
  Evaluate Franke's function for given x, y mesh.


  Parameters
  ----------
  x: n-dimensional array of floats
    Meshgrid for x.
  y: n-dimensional array of floats
    Meshgrid for y.


  Returns
  -------
  array like:
    Franke's function mesh.


  """
  term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
  term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
  term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
  term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
  return term1 + term2 + term3 + term4

def features_polynomial_2d(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """
    Construct design matrix for two-dimensional polynomial, where columns are:
    (1 + y + ... y**p) + x(1 + y + ... y**p-1) + ... + x**p,
    where p is degree for polynomial, x = x_i and y = y_j, indexed such that
    row index k  =  len(y)*i + j.


    Parameters
    ----------
    x: np.ndarray
        x inputs
    y: np.ndarray
        y inputs.
    degree : int
        Polynomial degree for model.


    Returns
    -------
    np.ndarray, shape (m*n, (degree+1)*(degree+2)/2), dtype float
        Design matrix for two dimensional polynomial of specified degree. 
    """
    len_x = x.shape[0]
    len_y = y.shape[0]
    features_xy = np.empty(
        (len_x*len_y, int((degree+1)*(degree+2)/2)), dtype=float)
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            row = len_y*i + j
            col_count = 0
            for k in range(degree + 1):
                for l in range(degree + 1 - k):
                    features_xy[row, col_count] = x_**k*y_**l
                    col_count += 1
    return features_xy

def generate_data_and_init_parameters():
    """Generate Franke Function mesh with noise
    """
    rng = np.random.default_rng(2023)
    x = rng.random(50)
    y = rng.random(50)
    x = np.linspace(0,1,50)
    y = np.linspace(0,1,50)
    x_mesh, y_mesh = np.meshgrid(x, y)
    noise = rng.normal(size=(50,50))*0.1
    target_mesh = franke_function(x_mesh, y_mesh) + noise
    target = target_mesh.flatten()
    # Generate feature polynomial
    degree = 10
    features = features_polynomial_2d(x, y, degree)
    # parameters and model
    n_parameters = int((degree+1)*(degree+2)/2)
    init_parameters = rng.random(n_parameters)
    return features, target, init_parameters

def gradient_descent():
    """Apply GD on model for Franke data with noise, with tuned learning rate.
    """
    features, target, init_parameters = generate_data_and_init_parameters()
    config = SystemConfig(features, target, cost_grad_func, init_parameters, 2023)
    optimizer = GradientDescent(config, 10e-6, 0)
    def meta_mse(parameters):
        model = features @ parameters
        return mean_squared_error(target, model)
    optimized_parameters = optimizer(10_000, meta_mse=meta_mse)
    best_model = features @ optimized_parameters
    mse_gd = mean_squared_error(target, best_model)
    print(optimizer)
    print(f"MSE: {mse_gd:.4g}")

def gradient_descent_with_momentum():
    """Apply DGM on model for Franke data with noise, with tuned learning rate.
    """
    features, target, init_parameters = generate_data_and_init_parameters()
    config = SystemConfig(features, target, cost_grad_func, init_parameters, 2023)
    optimizer = GradientDescent(config, 0.81e-6, 0.9)
    def meta_mse(parameters):
        model = features @ parameters
        return mean_squared_error(target, model)
    optimized_parameters = optimizer(10_000, meta_mse=meta_mse)
    best_model = features @ optimized_parameters
    mse_gd = mean_squared_error(target, best_model)
    print(optimizer)
    print(f"MSE: {mse_gd:.4g}")

def stochastic_gradient_descent():
    """apply SGD on model for Franke data with noise, with tuned learning rate
    """
    features, target, init_parameters = generate_data_and_init_parameters()
    config = SystemConfig(features, target, cost_grad_func, init_parameters, 2023)
    optimizer = StochasticGradientDescent(config, 7e-6, 0, target.shape[0]//64, 64)
    def meta_mse(parameters):
        model = features @ parameters
        return mean_squared_error(target, model)
    optimized_parameters = optimizer(10_000, meta_mse=meta_mse)
    best_model = features @ optimized_parameters
    mse_gdm = mean_squared_error(target, best_model)
    print(optimizer)
    print(f"MSE: {mse_gdm:.4g}")


def stochastic_gradient_descent_varying_minibatch_size():
    """apply SGD as specified in `stochastic_gradient_descent` with varying
    mini bath sizes.
    """
    features, target, init_parameters = generate_data_and_init_parameters()
    config = SystemConfig(features, target, cost_grad_func, init_parameters, 2023)
    mini_batch_sizes = [1,2,4,8,16,32,64,128]
    for mini_batch_size in mini_batch_sizes:
        optimizer = StochasticGradientDescent(config, 7e-6, 0, target.shape[0]//mini_batch_size, mini_batch_size)
        def meta_mse(parameters):
            model = features @ parameters
            return mean_squared_error(target, model)
        optimized_parameters = optimizer(10_000, meta_mse=meta_mse)
        best_model = features @ optimized_parameters
        mse_gdm = mean_squared_error(target, best_model)
        print(optimizer)
        print(f"MSE: {mse_gdm:.4g}")

def stochastic_gradient_descent_with_momentum():
    """Test Stochastic Gradient Descent
    """
    features, target, init_parameters = generate_data_and_init_parameters()
    config = SystemConfig(features, target, cost_grad_func, init_parameters, 2023)
    optimizer = StochasticGradientDescent(config, 0.86e-6, 0.9, target.shape[0]//64, 64)
    def meta_mse(parameters):
        model = features @ parameters
        return mean_squared_error(target, model)
    optimized_parameters = optimizer(10_000, meta_mse=meta_mse)
    best_model = features @ optimized_parameters
    mse_gdm = mean_squared_error(target, best_model)
    print(optimizer)
    print(f"MSE: {mse_gdm:.4g}")

def adagrad():
    """Test adagrad
    """
    features, target, init_parameters = generate_data_and_init_parameters()
    config = SystemConfig(features, target, cost_grad_func, init_parameters, 2023)
    optimizer = Adagrad(config, 0.06, 64)
    def meta_mse(parameters):
        model = features @ parameters
        return mean_squared_error(target, model)
    optimized_parameters = optimizer(10_000, meta_mse=meta_mse)
    model = features @ optimized_parameters
    adagrad_mse = mean_squared_error(target, model)
    print(optimizer)
    print(f"MSE: {adagrad_mse:.4g}")


def rmsprop():
    """Test RMSProp
    """
    features, target, init_parameters = generate_data_and_init_parameters()
    config = SystemConfig(features, target, cost_grad_func, init_parameters, 2023)
    optimizer = RMSProp(config, 900e-6, 64, 0.99982)
    def meta_mse(parameters):
        model = features @ parameters
        return mean_squared_error(target, model)
    optimized_parameters = optimizer(10_000, meta_mse=meta_mse)
    model = features @ optimized_parameters
    adagrad_mse = mean_squared_error(target, model)
    print(optimizer)
    print(f"MSE: {adagrad_mse:.4g}")

def adam():
    """
    Test ADAM
    """
    features, target, init_parameters = generate_data_and_init_parameters()
    config = SystemConfig(features, target, cost_grad_func, init_parameters, 2023)
    optimizer = ADAM(config, 1.64e-2, 64)
    def meta_mse(parameters):
        model = features @ parameters
        return mean_squared_error(target, model)
    optimized_parameters = optimizer(10_000, meta_mse=meta_mse)
    model = features @ optimized_parameters
    adagrad_mse = mean_squared_error(target, model)
    print(optimizer)
    print(f"MSE: {adagrad_mse:.4g}")


if __name__ == "__main__":
    # gradient_descent()
    # gradient_descent_with_momentum()
    # stochastic_gradient_descent()
    stochastic_gradient_descent_varying_minibatch_size()
    # stochastic_gradient_descent_with_momentum()
    # adagrad()
    # rmsprop()
    # adam()
















# def meta_loss(parameters: tuple, forward_pass_func: callable, loss_func: callable,
#               features: np.ndarray, target: np.ndarray):
#     """Compute loss of model w.r.t target.

#     Parameters
#     ----------
#     parameters : tuple of float
#         Parameters for FFNN.
#     forward_pass_func : callable
#         Callable representing the forward pass of the neural network.
#         It is expected to be a callable representing the `meta_forward_pass` method of 
#         the `FeedForwardNeuralNetwork` class.
#     loss_func : {mean_squared_error, cross_entropy}
#         Loss function.
#     features : np.ndarray
#         Features.
#     target : np.ndarray
#         Target.
#     """
#     if not isinstance(parameters, tuple):
#         raise TypeError(f"parameters: expected tuple, got {type(parameters)}.")
#     # for param in parameters:
#     #     if not isinstance(param, float):
#     #         raise TypeError(f"Each element in parameters should be a float, but found {type(param)}.")
#     if not callable(forward_pass_func):
#         raise TypeError(f"forward_pass_func: expected callable, got {type(forward_pass_func)}.")
#     if not callable(loss_func):
#         raise TypeError(f"loss_func: expected callable, got {type(loss_func)}.")
#     if not isinstance(features, (np.ndarray, jnp.ndarray)):
#         raise TypeError(f"features: expected np.ndarray, got {type(features)}.")
#     if not isinstance(target,( np.ndarray, jnp.ndarray)):
#         raise TypeError(f"target: expected np/jnp.ndarray, got {type(target)}.")

#     output = forward_pass_func(parameters, features)
#     return loss_func(target, output)



# def grad_meta_loss(parameters: tuple, meta_loss_func: callable):
#     """
#     compute grad of meta_loss w.r.t parameters
#     """
#     if not isinstance(parameters, tuple):
#         raise TypeError(f"parameters: expected tuple, got {type(parameters)}.")
#     # for param in parameters:
#     #     if not isinstance(param, float):
#     #         raise TypeError(f"Each element in parameters should be a float, but found {type(param)}.")
#     grad_func = grad(meta_loss_func)
#     return jnp.array(grad_func(parameters))



# def simple_ffnn_xor():
#   y_xor = jnp.array([0,1,1,1])
#   features = np.array([[0, 0],
#                      [0, 1],
#                      [1, 0],
#                      [1, 1]])
#   features = features.T
#   network = FeedForwardNeuralNetwork(2,1,2,1, sigmoid)
#   network.initialize_weights_and_biases()
#   parameters = network.extract_sparameters()
#   learning_rate = 0.1
#   tolerance = 1e-5
#   max_iter = 1000
#   meta_mse = lambda params: meta_loss(params, network.meta_forward_pass, 
#                                       mean_squared_error, features, y_xor)
#   grad_meta_mse = lambda params: grad_meta_loss(params, meta_mse)
#   optimizer = GradientDescent(learning_rate, grad_meta_mse, parameters)
#   trained_parameters, max_iter_met, converged = optimizer(max_iter, tolerance)
#   network.insert_parameters(trained_parameters)
#   output = network(features)
#   print(output)

# if __name__ == "__main__":
#     simple_ffnn_xor()