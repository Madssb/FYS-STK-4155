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
    """Generate Franke Function mesh with noise"""
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
    """Apply GD on model for Franke data with noise, with tuned learning rate.
    """
    learning_rates = np.logspace(-6,-4,100)
    convergence_epoch = np.empty_like(learning_rates)
    mse = np.empty_like(learning_rates)
    for i, learning_rate in enumerate(learning_rates):
        problem, convergence, target, features = problem_n_convergence(1e-1)
        optimizer = GradientDescent(problem, convergence, learning_rate, 0)
        optimized_parameters = optimizer(10_000)
        best_model = features @ optimized_parameters
        mse[i] = mean_squared_error(target, best_model)
        convergence_epoch[i] = optimizer.iteration
    results = pd.DataFrame({
        'Learning Rate': learning_rates,
        'Convergence Epoch': convergence_epoch,
        'Mean Squared Error': mse
    })
    fig = plt.figure(figsize=my_figsize())
    plt.scatter(learning_rates, convergence_epoch, c='black', s=1)
    plt.xlabel("Learning rate $\eta$")
    plt.xscale("log")
    plt.ylabel("Convergence epoch")
    fig.tight_layout()
    plt.show()
    fig.savefig("figures/regression/gd.pdf")
    min_index = np.argmin(convergence_epoch)
    print(f"most efficient learning rate: {learning_rates[min_index]},")
    print(f"converged in {convergence_epoch[min_index]} e-pochs.")

def gd_show_divergence():
    """Demonstrate MSE exploding when learning rate is too big
    """
    learning_rates = np.linspace(90e-6,170e-6,10)
    convergence_epoch = np.empty_like(learning_rates)
    mse = np.empty_like(learning_rates)
    for i, learning_rate in enumerate(learning_rates):
        problem, convergence, target, features = problem_n_convergence(1e-1,np.inf)
        optimizer = GradientDescent(problem, convergence, learning_rate, 0)
        optimized_parameters = optimizer(10_00)
        best_model = features @ optimized_parameters
        mse[i] = mean_squared_error(target, best_model)
        convergence_epoch[i] = optimizer.iteration
    results = pd.DataFrame({
        'Learning Rate': learning_rates,
        'Convergence Epoch': convergence_epoch,
        'Mean Squared Error': mse
    })
    fig = plt.figure(figsize=my_figsize())
    plt.scatter(learning_rates, mse, c='black', s=1)
    plt.xlabel("Learning rate $\eta$")
    plt.xscale("log")
    plt.ylabel("MSE")
    fig.tight_layout()
    plt.show()


def gradient_descent_with_momentum():
    """Apply GDM on model
    """
    learning_rates = np.logspace(-6, -4, 5)
    print(learning_rates)
    momentum_parameters = np.linspace(0, 0.9, 5)
    mses = np.empty((learning_rates.shape[0], momentum_parameters.shape[0]))
    convergence_epoch = np.empty_like(mses, dtype=int)

    for i, learning_rate in enumerate(learning_rates):
        for j, momentum_parameter in enumerate(momentum_parameters):
            problem, convergence, target, features = problem_n_convergence(1e-1)
            optimizer = GradientDescent(problem, convergence, learning_rate, momentum_parameter)
            optimized_parameters = optimizer(10_000)
            best_model = features @ optimized_parameters
            mses[i, j] = mean_squared_error(target, best_model)
            convergence_epoch[i, j] = optimizer.iteration

    # Create a DataFrame from the convergence epochs values
    results = pd.DataFrame(convergence_epoch, index=learning_rates, columns=momentum_parameters)

    # Create a customized heatmap using Seaborn
    plt.figure(figsize=(12, 8))  # Increase figure size
    sns.set(font_scale=1.2)  # Adjust font size
    sns.heatmap(results, cmap='coolwarm', annot=True, fmt="d", cbar=True, linewidths=.5, square=True)

    # Customize the tick labels on the y-axis (Learning Rate)
    ax = plt.gca()
    yticks = ax.get_yticks()
    ytick_labels = [f"{rate:.2g}" for rate in learning_rates]
    ax.set_yticklabels(ytick_labels)
    plt.xlabel('Momentum Parameter')
    plt.ylabel('Learning Rate')
    plt.title('Gradient Descent with Momentum Hyperparameter Tuning')
    plt.tight_layout()
    plt.savefig("figures/regression/gdm.pdf")


def stochastic_gradient_descent():
    """apply SGD on model for Franke data with noise, with tuned learning rate
    """
    problem, convergence, target, features = problem_n_convergence()
    optimizer = StochasticGradientDescent(problem, convergence, 7e-6, 0, 64)
    optimized_parameters = optimizer(10_000)
    best_model = features @ optimized_parameters
    mse_gd = mean_squared_error(target, best_model)
    print(optimizer)
    print(f"MSE: {mse_gd:.4g}")


def stochastic_gradient_descent_varying_minibatch_size():
    """apply SGD as specified in `stochastic_gradient_descent` with varying
    mini bath sizes.
    """
    problem, convergence, target, features = problem_n_convergence(5e-2)
    mini_batch_sizes = [1, 2, 4, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129]
    convergence_epoch = []
    mses = []
    cpu_times = []
    print(convergence)
    for mini_batch_size in mini_batch_sizes:
        problem, convergence, target, features = problem_n_convergence()
        optimizer = StochasticGradientDescent(problem, convergence, 0.4e-4, 0, mini_batch_size)
        start_time = time.process_time()
        optimized_parameters = optimizer(10_000)
        end_time = time.process_time()
        best_model = features @ optimized_parameters
        mse_gd = mean_squared_error(target, best_model)
        convergence_epoch.append(optimizer.iteration)
        mses.append(mse_gd)
        cpu_times.append(end_time - start_time)
        print(optimizer)
        print(f"MSE: {mse_gd:.4g}")
    mini_batch_size_arr = np.array(mini_batch_sizes)
    convergence_iteration_arr = np.array(convergence_epoch)
    data = list(zip(mini_batch_sizes, convergence_epoch, cpu_times))
    print(tabulate(data, headers=["mini batch size", "epoch of convergence", "cpu time"], tablefmt="latex"))

def stochastic_gradient_descent_with_momentum():
    """Test Stochastic Gradient Descent"""
    problem, convergence, target, features = problem_n_convergence(1e-2)
    learning_rates = np.logspace(-6, -5, 5)
    momentum_parameters = np.linspace(0,0.9,5) 
    mses = np.empty((5, 5))
    mses_list = []
    
    for learning_rate in learning_rates:
        for momentum_parameter in momentum_parameters:
            try:
                optimizer = StochasticGradientDescent(problem, convergence, learning_rate, momentum_parameter, 128)
                optimized_parameters = optimizer(1_000)
                best_model = features @ optimized_parameters
                mse = mean_squared_error(target, best_model)
                
                # Append the results to the list
                mses_list.append({"Learning Rate": learning_rate, "Momentum Parameter": momentum_parameter, "MSE": mse})
                
                # Print the result
                print(optimizer)
                print(f"MSE: {mse:.4g}")
            except ValueError:
                # If optimization diverged, you can skip this iteration
                continue
    
    # Create a DataFrame from the list of results
    mses_df = pd.DataFrame(mses_list)

    # Pivot the DataFrame to create a 2D table for the heatmap
    heatmap_data = mses_df.pivot_table(index="Learning Rate", columns="Momentum Parameter", values="MSE")

    # Create the heatmap using Seaborn
    plt.figure()
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".4f", linewidths=0.5)
    plt.xlabel("Momentum Parameter")
    plt.ylabel("Learning Rate")

    # Set the number of digits on the axes and add labels
    plt.xticks(np.arange(len(heatmap_data.columns)), heatmap_data.columns.map(lambda x: f"{x:.4f}"), rotation=45)
    plt.yticks(np.arange(len(heatmap_data.index)), heatmap_data.index.map(lambda x: f"{x:.4f}"))
    plt.tight_layout()
    plt.show()

def adagrad():
    """Test adagrad"""
    problem, convergence, target, features = problem_n_convergence()
    optimizer = Adagrad(problem, convergence, 0.06, 128)
    optimized_parameters = optimizer(10_000)
    best_model = features @ optimized_parameters
    mse_gd = mean_squared_error(target, best_model)
    print(optimizer)
    print(f"MSE: {mse_gd:.4g}")


def rmsprop():
    """Test RMSProp"""
    problem, convergence, target, features = problem_n_convergence()
    optimizer = RMSProp(problem, convergence, 900e-6, 128, 0.99982)
    optimized_parameters = optimizer(10_000)
    best_model = features @ optimized_parameters
    mse_gd = mean_squared_error(target, best_model)
    print(optimizer)
    print(f"MSE: {mse_gd:.4g}")


def adam():
    """
    Test ADAM
    """
    problem, convergence, target, features = problem_n_convergence()
    optimizer = ADAM(problem, convergence, 1.64e-2, 128)
    optimized_parameters = optimizer(10_000)
    best_model = features @ optimized_parameters
    mse_gd = mean_squared_error(target, best_model)
    print(optimizer)
    print(f"MSE: {mse_gd:.4g}")

def adam2():
    """
    Test ADAM
    """
    mini_batch_sizes = [2,4,8,9,16,17,32,33,64,65,128,129]
    convergence_epoch = []
    cpu_times = []
    for mini_batch_size in mini_batch_sizes:
        problem, convergence, target, features = problem_n_convergence()
        optimizer = ADAM(problem, convergence, 1.64e-2, mini_batch_size)
        start = time.process_time()
        optimized_parameters = optimizer(10_000)
        end = time.process_time()
        cpu_times.append(end - start)
        convergence_epoch.append(optimizer.iteration)
        best_model = features @ optimized_parameters
    data = list(zip(mini_batch_sizes, convergence_epoch, cpu_times))
    print(tabulate(data, headers=["mini batch size", "epoch of convergence", "cpu time"]))


if __name__ == "__main__":
    # gradient_descent()
    # gradient_descent_with_momentum()
    gd_show_divergence()
    # stochastic_gradient_descent()
    # stochastic_gradient_descent_varying_minibatch_size()
    # stochastic_gradient_descent_with_momentum()
    # adagrad()
    # rmsprop()
    # adam()
    # adam2()