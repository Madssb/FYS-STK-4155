import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from jax import grad
# from ffnn import FeedForwardNeuralNetwork
from nn_class import FeedForwardNeuralNetwork
from FrankeFunction import franke_function
from SGD import SGD_const, SGD_AdaGrad, SGD_RMSProp, SGD_ADAM
from nn_class import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative, identity
from nn_class import hard_classifier, indicator, accuracy_score, MSE, R2
from sklearn.model_selection import train_test_split
# from gradient_descents import GradientDescent
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

def nn_regression_network_OLS(learning_method=SGD_const):
    points = 100
    x = np.arange(0, 1, 1/points)
    y = np.arange(0, 1, 1/points)
    z = franke_function(x, y)
    X = np.array([x, y]).T

    etas = np.logspace(0, -5, 6)
    # l2 = np.logspace(0, -5, 6)
    neurons = [5, 10, 30, 50]
    layers = np.linspace(1, 3, 3)
    batch_sizes = [2**i for i in range(4, 8)]

    MSE_lay_neur = np.zeros((len(layers),len(neurons)))
    R2_lay_neur = np.zeros((len(layers),len(neurons)))

    # constant learning rate

    for i, l in enumerate(layers):
        for j, n in enumerate(neurons):
            nn = FeedForwardNeuralNetwork(X, z, n_hidden_layers=int(l), n_hidden_neurons=n, L2=0,
                                          output_activation_function=identity,
                                          hidden_activation_function=sigmoid,
                                          hidden_activation_derivative=sigmoid_derivative)
            nn.train(learning_method, n_epochs=300, init_lr=0.1, batch_size=len(z))
            MSE_lay_neur[i, j] = MSE(z, nn.predict(X))
            R2_lay_neur[i, j] = R2(z, nn.predict(X))
    
    MSE_scores = pd.DataFrame(MSE_lay_neur, columns = neurons, index = layers)
    R2_scores = pd.DataFrame(MSE_lay_neur, columns = neurons, index = layers)

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(MSE_scores, annot=True, ax=ax, cmap="viridis")
    ax.set_title(f'SGD_const MSE')
    ax.set_xlabel("$neurons$")
    ax.set_ylabel("$layers$")

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(R2_scores, annot=True, ax=ax, cmap="viridis")
    ax.set_title(f'SGD_const R2')
    ax.set_xlabel("$neurons$")
    ax.set_ylabel("$layers$")

    plt.show()



    # plt.savefig('figures/nn_classification/train_accuracy_ReLU.pdf')
    # plt.savefig('figures/nn_classification/train_accuracy_ReLU.png')


if __name__ == "__main__":
    # simple_ffnn_xor()
    nn_regression_network_OLS(learning_method=SGD_const)