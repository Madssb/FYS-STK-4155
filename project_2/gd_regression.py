"""
TO DO LIST


1. Vis hva enn som går ann å vises med plain GD (X)
1.1. vis sammenhengen med økning i learning rate, og antall e-pochs krevd for konvergens (X)
1.2. vis at ved for høye learning rates så eksploderer MSEen (X)

2. vis hva enn som gar ann å vise med GDM
2.1. vis sammenhengen mellom learning rate, og momentum parameter og antall e-pochs krevd for konvergens.

3. plukk ut best performing 
3.1. utforsk MSE sin sammenheng med mini-batch size
3.2. 
"""
import numpy as np
from gradient_descents import *
from loss_functions import mean_squared_error
from tabulate import tabulate
import time
import pandas as pd
import seaborn as sns
from utilities import franke_function, my_figsize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
        design_matrix[:, i] = input**i

    return design_matrix


def cost_grad_func(features, target, parameters):
    """
    cost function differentiated w.r.t parameters
    """
    return -2 * features.T @ (target - features @ parameters)


def grad_func():
    pass

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
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
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
        (len_x * len_y, int((degree + 1) * (degree + 2) / 2)), dtype=float
    )
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            row = len_y * i + j
            col_count = 0
            for k in range(degree + 1):
                for l in range(degree + 1 - k):
                    features_xy[row, col_count] = x_**k * y_**l
                    col_count += 1
    return features_xy


def problem_n_convergence(tolerance, divergence_tol=1e3):
    """Generate Franke Function mesh with noise
    """
    rng = np.random.default_rng(2023)
    x = rng.random(50)
    y = rng.random(50)
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    x_mesh, y_mesh = np.meshgrid(x, y)
    noise = rng.normal(size=(50, 50)) * 0.1
    target_mesh = franke_function(x_mesh, y_mesh) + noise
    target = target_mesh.flatten()
    # Generate feature polynomial
    degree = 10
    features = features_polynomial_2d(x, y, degree)
    # parameters and model
    n_parameters = int((degree + 1) * (degree + 2) / 2)
    init_parameters = rng.random(n_parameters)
    features_train, features_test, target_train, target_test = train_test_split(features, target)
    def meta_mse(parameters):
        model = features_train @ parameters
        return mean_squared_error(target_train, model)
    problem = ProblemConfig(features_train, target_train, cost_grad_func, init_parameters, 2023)
    convergence = ConvergenceConfig(meta_mse, tolerance, divergence_tol)
    return problem, convergence, target_test, features_test


def gradient_descent():
    """apply GD on franke func; tweak initial learning rate for fastest convergence.
    """
    learning_rates = np.logspace(-6,-4,100)
    convergence_epoch = np.empty_like(learning_rates, dtype=int)
    mse = np.empty_like(learning_rates)
    for i, learning_rate in enumerate(learning_rates):
        problem, convergence, target, features = problem_n_convergence(1e-1)
        optimizer = GradientDescent(problem, convergence, learning_rate, 0)
        optimized_parameters = optimizer(10_000)
        best_model = features @ optimized_parameters
        mse[i] = mean_squared_error(target, best_model)
        convergence_epoch[i] = optimizer.iteration
    fig, ax = plt.subplots(figsize=my_figsize())
    ax.scatter(learning_rates, convergence_epoch, c='black', s=1)
    ax.set_xticks(learning_rates, labels=[f"{learning_rate:.3g}" for learning_rate in learning_rates])
    ax.set_xlabel("Learning rate $\eta$")
    ax.set_xscale("log")
    ax.set_ylabel("Convergence epoch")
    fig.tight_layout()
    fig.savefig("figures/regression/gd.pdf")
    min_index = np.argmin(convergence_epoch)
    best_convergence_epoch = convergence_epoch[min_index]
    best_learning_rate = learning_rates[min_index]
    return best_convergence_epoch, best_learning_rate
    

def gd_show_divergence():
    """Demonstrate MSE exploding when learning rate is too big
    """
    learning_rates = np.linspace(90e-6,170e-6,10)
    convergence_epoch = np.empty_like(learning_rates, dtype=int)
    mse = np.empty_like(learning_rates)
    for i, learning_rate in enumerate(learning_rates):
        problem, convergence, target, features = problem_n_convergence(1e-1,np.inf)
        optimizer = GradientDescent(problem, convergence, learning_rate, 0)
        optimized_parameters = optimizer(10_00)
        best_model = features @ optimized_parameters
        mse[i] = mean_squared_error(target, best_model)
        convergence_epoch[i] = optimizer.iteration
    fig, ax = plt.subplots(figsize=my_figsize())
    ax.scatter(learning_rates, convergence_epoch, c='black', s=1)
    ax.set_xscale("log")
    ax.set_xticks(learning_rates, labels=[f"{learning_rate:.3g}" for learning_rate in learning_rates], rotation=45)
    ax.set_xlabel("Learning rate $\eta$")
    ax.set_ylabel("Convergence epoch")
    fig.tight_layout()
    fig.savefig("figures/regression/divergence.pdf")


def gradient_descent_with_momentum():
    """Apply GDM on model, search for fastest convergence
    """
    learning_rates = np.logspace(-6, -4, 5)
    momentum_params = np.linspace(0, 0.9, 5)
    mses = np.empty((learning_rates.shape[0], momentum_params.shape[0]))
    convergence_epoch = np.empty_like(mses, dtype=int)

    for i, learning_rate in enumerate(learning_rates):
        for j, momentum_param in enumerate(momentum_params):
            problem, convergence, target, features = problem_n_convergence(1e-1)
            optimizer = GradientDescent(problem, convergence, learning_rate, momentum_param)
            optimized_parameters = optimizer(10_000)
            best_model = features @ optimized_parameters
            mses[i, j] = mean_squared_error(target, best_model)
            convergence_epoch[i, j] = optimizer.iteration
    # Create a DataFrame from the convergence epochs values
    results = pd.DataFrame(convergence_epoch, index=learning_rates, columns=momentum_params)
    fig, ax = plt.subplots()
    sns.heatmap(results, cmap='coolwarm', annot=True, fmt="d", cbar=True, linewidths=.5, square=True,
                cbar_kws={'label': 'Convergence e-poch'},
                xticklabels=[f"{momentum_param:.3g}" for momentum_param in momentum_params],
                yticklabels=[f"{learning_rate:.3g}" for learning_rate in learning_rates])
    ax.set_xlabel('Momentum Parameter')
    ax.set_ylabel('Learning Rate')
    fig.tight_layout()
    fig.savefig("figures/regression/gdm.pdf")
    min_index_flat = np.argmin(convergence_epoch)
    min_indices = np.unravel_index(min_index_flat, convergence_epoch.shape)
    best_convergence_epoch = convergence_epoch[min_indices]
    best_learning_rate = learning_rates[min_indices[0]]
    best_momentum_param = momentum_params[min_indices[1]]
    return best_convergence_epoch, best_learning_rate, best_momentum_param



def stochastic_gradient_descent():
    """apply SGD on model for Franke data with noise, with tuned learning rate
    """
    # problem, convergence, target, features = problem_n_convergence()
    # optimizer = StochasticGradientDescent(problem, convergence, 7e-6, 0, 64)
    # optimized_parameters = optimizer(10_000)
    # best_model = features @ optimized_parameters
    # mse_gd = mean_squared_error(target, best_model)
    # print(optimizer)
    # print(f"MSE: {mse_gd:.4g}")
    learning_rates = np.logspace(-6,-4,100)
    convergence_epoch = np.empty_like(learning_rates, dtype=int)
    mse = np.empty_like(learning_rates)
    for i, learning_rate in enumerate(learning_rates):
        problem, convergence, target, features = problem_n_convergence(1e-1)
        optimizer = StochasticGradientDescent(problem, convergence, learning_rate, 0, 128)
        optimized_parameters = optimizer(10_000)
        best_model = features @ optimized_parameters
        mse[i] = mean_squared_error(target, best_model)
        convergence_epoch[i] = optimizer.iteration
    fig, ax = plt.subplots(figsize=my_figsize())
    ax.scatter(learning_rates, convergence_epoch, c='black', s=1)
    ax.set_xticks(learning_rates, labels=[f"{learning_rate:.3g}" for learning_rate in learning_rates])
    ax.set_xlabel("Learning rate $\eta$")
    ax.set_ylabel("Convergence epoch")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig("figures/regression/sgd.pdf")
    min_index = np.argmin(convergence_epoch)
    best_convergence_epoch = convergence_epoch[min_index]
    best_learning_rate = learning_rates[min_index]
    return best_convergence_epoch, best_learning_rate


def stochastic_gradient_descent_varying_minibatch_size():
    """apply SGD as specified in `stochastic_gradient_descent` with varying
    mini bath sizes.
    """
    mini_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    convergence_epoch = []
    mses = []
    cpu_times = []
    for mini_batch_size in mini_batch_sizes:
        problem, convergence, target, features = problem_n_convergence(1e-1)
        optimizer = StochasticGradientDescent(problem, convergence, 9.11e-5, 0, mini_batch_size)
        start_time = time.process_time()
        optimized_parameters = optimizer(10_000)
        end_time = time.process_time()
        best_model = features @ optimized_parameters
        mse_gd = mean_squared_error(target, best_model)
        convergence_epoch.append(optimizer.iteration)
        mses.append(mse_gd)
        cpu_times.append(end_time - start_time)
    data = list(zip(mini_batch_sizes, convergence_epoch, cpu_times))
    print(tabulate(data, headers=["mini batch size", "epoch of convergence", "cpu time"], tablefmt="latex"))

def stochastic_gradient_descent_with_momentum():
    """Test Stochastic Gradient Descent
    """

    learning_rates = np.logspace(-6, -5, 5)
    momentum_params = np.linspace(0,0.9,5) 
    mse = np.empty((5, 5), dtype=float)
    convergence_epoch = np.empty_like(mse, dtype=int)
    for i, learning_rate in enumerate(learning_rates):
        for j, momentum_param in enumerate(momentum_params):
            problem, convergence, target, features = problem_n_convergence(1e-1)
            optimizer = StochasticGradientDescent(problem, convergence, learning_rate, momentum_param, 128)
            optimized_parameters = optimizer(10_000)
            best_model = features @ optimized_parameters
            mse[i, j] = mean_squared_error(target, best_model)
            convergence_epoch[i, j] = optimizer.iteration
            print(i+j)
    results = pd.DataFrame(convergence_epoch, index=learning_rates, columns=momentum_params)
    fig,ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(results, cmap='coolwarm', annot=True, fmt="d", cbar=True, linewidths=.5, square=True,
                cbar_kws={'label': 'Convergence e-poch'},
                xticklabels=[f"{momentum_param:.3g}" for momentum_param in momentum_params],
                yticklabels=[f"{learning_rate:.3g}" for learning_rate in learning_rates])
    ax.set_xlabel('Momentum Parameter')
    ax.set_ylabel('Learning Rate')
    fig.tight_layout()
    plt.savefig("figures/regression/sgdm.pdf")
    min_index_flat = np.argmin(convergence_epoch)
    min_indices = np.unravel_index(min_index_flat, convergence_epoch.shape)
    best_convergence_epoch = convergence_epoch[min_indices]
    best_learning_rate = learning_rates[min_indices[0]]
    best_momentum_param = momentum_params[min_indices[1]]
    return best_convergence_epoch, best_learning_rate, best_momentum_param


def adagrad():
    """Test adagrad
    """
    learning_rates = np.logspace(1,-1,100)
    convergence_epoch = np.empty_like(learning_rates, dtype=int)
    mse = np.empty_like(learning_rates)
    for i, learning_rate in enumerate(learning_rates):
        problem, convergence, target, features = problem_n_convergence(1e-1)
        optimizer = Adagrad(problem, convergence, learning_rate, 128)
        optimized_parameters = optimizer(10_000)
        best_model = features @ optimized_parameters
        mse[i] = mean_squared_error(target, best_model)
        convergence_epoch[i] = optimizer.iteration
        print(i)
    fig, ax = plt.subplots(figsize=my_figsize())
    ax.scatter(learning_rates, convergence_epoch, c='black', s=1)
    ax.set_xticks(learning_rates, labels=[f"{learning_rate:.3g}" for learning_rate in learning_rates])
    ax.set_xlabel("Learning rate $\eta$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Convergence epoch")
    fig.tight_layout()
    fig.savefig("figures/regression/adagrad.pdf")
    min_index = np.argmin(convergence_epoch)
    best_convergence_epoch = convergence_epoch[min_index]
    best_learning_rate = learning_rates[min_index]
    return best_convergence_epoch, best_learning_rate

def rmsprop():
    """Test RMSProp
    """
    learning_rates = np.logspace(-2,0,5)
    decay_rates = np.linspace(0.5, 0.99, 5)
    convergence_epoch = np.empty((learning_rates.shape[0], decay_rates.shape[0]), dtype=int)
    mse = np.empty_like(convergence_epoch)
    counter = 0
    for i, learning_rate in enumerate(learning_rates):
        for j, decay_rate in enumerate(decay_rates):
            problem, convergence, target, features = problem_n_convergence(1e-1)
            optimizer = RMSProp(problem, convergence, learning_rate, 128, decay_rate)
            optimized_parameters = optimizer(1_000)
            best_model = features @ optimized_parameters
            mse[i, j] = mean_squared_error(target, best_model)
            convergence_epoch[i, j] = optimizer.iteration
            counter += 1
            print(counter)
    results = pd.DataFrame(convergence_epoch, index=learning_rates, columns=decay_rates)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(results, cmap='coolwarm', annot=True, fmt="d", cbar=True, linewidths=.5, square=True,
                cbar_kws={'label': 'Convergence e-poch'},
                xticklabels=[f"{decay_rate:.3g}" for decay_rate in decay_rates],
                yticklabels=[f"{learning_rate:.3g}" for learning_rate in learning_rates])
    ax.set_xlabel('decay rate')
    ax.set_ylabel('Learning Rate')
    fig.tight_layout()
    fig.savefig("figures/regression/rmsprop.pdf")
    min_index_flat = np.argmin(convergence_epoch)
    min_indices = np.unravel_index(min_index_flat, convergence_epoch.shape)
    best_convergence_epoch = convergence_epoch[min_indices]
    best_learning_rate = learning_rates[min_indices[0]]
    best_decay_rate = decay_rates[min_indices[1]]
    return best_convergence_epoch, best_learning_rate, best_decay_rate



def adam():
    """
    Test ADAM
    """
    learning_rates = np.logspace(-2,0,100)
    mse = np.empty_like(learning_rates, dtype=float)
    convergence_epoch = np.empty_like(mse, dtype=int)
    for i, learning_rate in enumerate(learning_rates):
        problem, convergence, target, features = problem_n_convergence(1e-1)
        optimizer = ADAM(problem, convergence, learning_rate, 128)
        optimized_parameters = optimizer(10_000)
        best_model = features @ optimized_parameters
        mse[i] = mean_squared_error(target, best_model)
        convergence_epoch[i] = optimizer.iteration
        print(i)
    fig, ax = plt.subplots(figsize=my_figsize())
    ax.scatter(learning_rates, convergence_epoch, c='black', s=1)
    ax.set_xticks(learning_rates, labels=[f"{learning_rate:.3g}" for learning_rate in learning_rates])
    ax.set_xlabel("Learning rate $\eta$")
    ax.set_ylabel("Convergence epoch")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig("figures/regression/adam.pdf")
    min_index = np.argmin(convergence_epoch)
    best_convergence_epoch = convergence_epoch[min_index]
    best_learning_rate = learning_rates[min_index]
    return best_convergence_epoch, best_learning_rate


def compare_gds():
    """Collect best performing optimizers and display as table. Additionally calls every single function.
    """
    optimizers = np.array(["GD", "GDM", "SGD", "SGDM", "Adagrad", "RMSProp", "ADAM"])
    funcs = [gradient_descent, gradient_descent_with_momentum, stochastic_gradient_descent, stochastic_gradient_descent_with_momentum, adagrad, rmsprop, adam]
    convergence_epochs = np.empty_like(optimizers, dtype=int)
    learning_rates = np.empty_like(optimizers, dtype=float)
    for idx, func in enumerate(funcs):
        tuple = func()
        convergence_epochs[idx] = tuple[0]
        learning_rates[idx] = tuple[1]
    table_data = zip(optimizers, convergence_epochs, learning_rates)
    print(tabulate(table_data, headers=["Optimizer", "Convergence Epochs", "Learning Rate"], tablefmt="latex"))


def cost_grad_func_ridge():
    """ehm
    """



if __name__ == "__main__":
    # gradient_descent()
    # gradient_descent_with_momentum()
    # gd_show_divergence()
    # stochastic_gradient_descent()
    # stochastic_gradient_descent_varying_minibatch_size()
    # stochastic_gradient_descent_with_momentum()
    # adagrad()
    # rmsprop()
    # adam()
    # adam2()
    # compare_gds()
    gd_show_divergence()
    plt.show()