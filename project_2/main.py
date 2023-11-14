import numpy as np
import jax.numpy as jnp
from jax import grad
import warnings
from ffnn import FeedForwardNeuralNetwork
from gradient_descents import StochasticGradientDescent, SGDConfig, Adagrad
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

# Generate data
rng = np.random.default_rng(2023)
x = rng.random(50)
y = rng.random(50)
# x = np.linspace(0,1,50)
# y = np.linspace(0,1,50)
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
model = features @ init_parameters
model_mesh = model.reshape(target_mesh.shape)
# Create a 3D surface plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_mesh, y_mesh, model_mesh, cmap='viridis')

# # Add labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# plt.title('3D Surface Plot of a 2D Polynomial')
# plt.show()

# learning_rate = 1e-4
# momentum = 0.9
# max_iter = 10_000
# tolerance = 5e-5
# config = SGDConfig(1e-4, None, target.shape[0]//64, 64, 2023)
# adagrad_optimizer = Adagrad(config, features, target, cost_grad_func, init_parameters)
# optimized_parameters = adagrad_optimizer(10_000,1e-5)
# model = features @ optimized_parameters
# adagrad_mse = mean_squared_error(target, model)
# print(f"AdaGrad MSE: {adagrad_mse:.4g}")
# quit()





learning_rate = 1e-4
momentum = 0.9
max_iter = 10_000
tolerance = 5e-5
# plain Gradient Descent
warnings.filterwarnings("ignore", category=RuntimeWarning)
def run_gradient_methods(learning_rate, momentum, max_iter, tolerance):
    print(f"{learning_rate=:.4g}, {momentum=:.4g}, {tolerance=:.4g}")
    config = SGDConfig(learning_rate, 0, 1, target.shape[0], 2023)
    optimizer = StochasticGradientDescent(config, features, target, cost_grad_func, init_parameters)
    best_parameters = optimizer(max_iter,tolerance)
    best_model = features @ best_parameters
    mse_gd = mean_squared_error(target, best_model)
    print(f"Gradient Descent MSE: {mse_gd:.4g}")
    # Gradient Descent with momentum
    config = SGDConfig(learning_rate, momentum, 1, target.shape[0], 2023)
    optimizer = StochasticGradientDescent(config, features, target, cost_grad_func, init_parameters)
    best_parameters = optimizer(max_iter,tolerance)
    best_model = features @ best_parameters
    mse_gdm = mean_squared_error(target, best_model)
    print(f"Gradient Descent with Momentum MSE: {mse_gdm:.4g}")
    # Stochastic Gradient Descent
    config = SGDConfig(learning_rate, 0, int(target.shape[0]/64),  16, 2023)
    optimizer = StochasticGradientDescent(config, features, target, cost_grad_func, init_parameters)
    best_parameters = optimizer(max_iter,tolerance)
    best_model = features @ best_parameters
    mse_gdm = mean_squared_error(target, best_model)
    print(f"Stochastic Gradient Descent MSE: {mse_gdm:.4g}")
    # Stochastic Gradient Descent with Momentum
    config = SGDConfig(learning_rate, momentum, int(target.shape[0]/64),  16, 2023)
    optimizer = StochasticGradientDescent(config, features, target, cost_grad_func, init_parameters)
    best_parameters = optimizer(max_iter,tolerance)
    best_model = features @ best_parameters
    mse_gdm = mean_squared_error(target, best_model)
    print(f"Stochastic Gradient Descent with Momentum MSE: {mse_gdm:.4g}")
    adagrad_optimizer = Adagrad(config, features, target, cost_grad_func, init_parameters)
    optimized_parameters = adagrad_optimizer(10_000,1e-5)
    model = features @ optimized_parameters
    adagrad_mse = mean_squared_error(target, model)
    print(f"AdaGrad MSE: {adagrad_mse:.4g}")
run_gradient_methods(103e-6, 0.9, 10_000, 1e-3)
#run_gradient_methods(4e-4, 0.9, 10_000, 1e-4)

def gd_optimize():
    """
    Optimize polynomial with plain gradient descent
    """
    print("Plain gradient descent")
    rng = np.random.default_rng(2023)
    input = rng.random(50)
    degree = 5
    #input = np.linspace(-10,10,50)
    poly_coeffs =10*rng.random(degree + 1)
    print(f"{poly_coeffs=}")
    poly  = np.polynomial.Polynomial(poly_coeffs)
    noise = rng.normal(50) 
    target = poly(input) + noise
    features = poly_features(input, degree)
    init_parameters = 10*rng.random(degree + 1)
    print(f"{init_parameters=}")
    model = features @ init_parameters
    config = SGDConfig(1e-4, 0, 1, input.shape[0], 2023)
    optimizer = StochasticGradientDescent(config, features, target, cost_grad_func, init_parameters)
    best_parameters = optimizer(100_000, 1e-8)
    print(f"{best_parameters=}")
    model = features @ best_parameters
    gd_mse = mean_squared_error(target, model)
    print(f"{gd_mse=:.4g}")
    
    plt.scatter(input,target, c='black', s=10, label="data")
    plt.scatter(input, model, c='blue', s=1, label="model")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()
    plt.show()

def gdm_optimize():
    print("gradient descent with momentum")
    rng = np.random.default_rng(2023)
    input = rng.random(50)
    degree = 5
    #input = np.linspace(-10,10,50)
    poly_coeffs =10*rng.random(degree + 1)
    print(f"{poly_coeffs=}")
    poly  = np.polynomial.Polynomial(poly_coeffs)
    noise = rng.normal(50) 
    target = poly(input) + noise
    features = poly_features(input, degree)
    init_parameters = 10*rng.random(degree + 1)
    print(f"{init_parameters=}")
    model = features @ init_parameters
    config = SGDConfig(1e-4, 0.9, 1, input.shape[0], 2023)
    optimizer = StochasticGradientDescent(config, features, target, cost_grad_func, init_parameters)
    best_parameters = optimizer(100_000, 1e-8)
    print(f"{best_parameters=}")
    model = features @ best_parameters
    gd_mse = mean_squared_error(target, model)
    print(f"{gd_mse=:.4g}")
    
    plt.scatter(input,target, c='black', s=10, label="data")
    plt.scatter(input, model, c='blue', s=1, label="model")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()
    plt.show()


def sgd_optimize():
    print("stochastic gradient descent")
    rng = np.random.default_rng(2023)
    input = rng.random(50)
    degree = 5
    #input = np.linspace(-10,10,50)
    poly_coeffs =10*rng.random(degree + 1)
    print(f"{poly_coeffs=}")
    poly  = np.polynomial.Polynomial(poly_coeffs)
    noise = rng.normal(50) 
    target = poly(input) + noise
    features = poly_features(input, degree)
    init_parameters = 10*rng.random(degree + 1)
    print(f"{init_parameters=}")
    model = features @ init_parameters
    config = SGDConfig(1e-4, 0.9, int(target.shape[0]/32),  32, 2023)
    optimizer = StochasticGradientDescent(config, features, target, cost_grad_func, init_parameters)
    best_parameters = optimizer(100_000, 1e-8)
    print(f"{best_parameters=}")
    model = features @ best_parameters
    gd_mse = mean_squared_error(target, model)
    print(f"{gd_mse=:.4g}")
    
    plt.scatter(input,target, c='black', s=10, label="data")
    plt.scatter(input, model, c='blue', s=1, label="model")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()
    plt.show()

# if __name__ == "__main__":
#     gd_optimize()
#     gdm_optimize()
#     sgd_optimize()




















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