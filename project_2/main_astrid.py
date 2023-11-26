"""
Solve project 2 here
"""

# Import external packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import warnings
import seaborn as sns

# Import self-made packages
from nn_class import FeedForwardNeuralNetwork
from nn_class import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative, identity
from nn_class import hard_classifier, indicator, accuracy_score, MSE, R2
from SGD import SGD_const, SGD_AdaGrad, SGD_RMSProp, SGD_ADAM
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from FrankeFunction import franke_function, features_polynomial_2d

def scaling(data_to_scale):
		scaler = StandardScaler()
		scaler.fit(data_to_scale)
		data_scaled = scaler.transform(data_to_scale)
		return data_scaled

# Plot formatting
sns.set_theme()
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'axes.grid': True})
plt.rc('legend', frameon=False)
params = {'legend.fontsize': 25,
			'figure.figsize': (12, 9),
			'axes.labelsize': 25,
			'axes.titlesize': 25,
			'xtick.labelsize': 'x-large',
			'ytick.labelsize': 'x-large',
     'font.size': 14,
     'axes.grid': True,
     'legend.frameon': False,}


def franke_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def simple_polynomial_GD(random_state=2023):
    
    # Fix random numbers
    rng = np.random.RandomState(random_state)

    # Analytical gradients

    def gradient_OLS(X, y, theta):
        N = len(y)
        return 2.0 / N * X.T @ ((X @ theta) - y)
    
    def gradient_Ridge(X, y, theta, L2):
        N = len(y)
        return X.T @ ((X @ theta) - y) +  L2 * (theta)
    
    # Convergence criteria
    tolerance = 1e-8

    # Generate a random polynomial
    n = 100
    x = rng.rand(n)#*np.linspace(-10, 10, n)
    #y = rng.rand(n)#*np.linspace(-10, 10, n)
    #z = franke_function(x, y)
    y = 4 + 3*x + x**2 + rng.randn(n)

    #X = np.array([np.ones(100), x, y, x**2, y**2, y*x**2, x*y**2, x**3, y**3]).T
    X = np.array([np.ones(100), x, x**2]).T

    theta = rng.rand(np.shape(X)[1])

    gradient_descent = SGD_const(X, y, gradient_OLS, init_model_parameters=theta, 
                                init_lr=0.0001, batch_size=n, momentum = 0, random_state=random_state)

    #gradient_descent_momentum = SGD_const(X, y, gradient_OLS, init_model_parameters=theta, 
    #                            init_lr=0.001, batch_size=n, momentum=0.9, random_state=random_state)
    
    for i in range(10000):
        new_theta = gradient_descent.advance(theta)
        if abs(new_theta - theta).all() < tolerance:
            print("Number of iterations before convergence: ", i)
            break
        theta = new_theta
    # print(MSE(y, X @ theta))
    # quit()
    for i in range(10):
        new_theta = gradient_descent_momentum.advance(theta)
        if abs(new_theta - theta).all() < tolerance:
            print("Number of iterations before convergence: ", i)
            break
    
    plt.scatter(x, y, s = 3)
    plt.scatter(x, X @ theta, label = 'GD', s = 5)
    #plt.scatter(x, X @ theta_momentum, label = 'GD Momentum', s = 7)
    plt.legend()
    plt.show()

def nn_regression(random_state=2023):

    # Fix random numbers
    rng = np.random.RandomState(random_state)

    # n = 100
    # # # #x = np.expand_dims(np.linspace(-10, 10, 100), 1)
    # x = rng.rand(n)
    # y = rng.rand(n)
    # x_mesh, y_mesh = np.meshgrid(x, y)
    # z = franke_function(x_mesh, y_mesh)
    
    points = 100
    x = np.arange(0, 1, 1/points)
    y = np.arange(0, 1, 1/points)
    # x_mesh, y_mesh = np.meshgrid(x, y)
    # z = franke_function(x_mesh, y_mesh)
    z = franke_function(x, y)
    # print(np.shape(z))
    # exit()
    # noise = np.random.normal(0, 1, x_mesh.shape)*0.1 # dampened noise
    # z = z + noise

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # surface = ax.plot_surface(x_mesh, y_mesh, z, cmap='viridis')
    # plt.plot(x, z)
    # plt.show()
    # # print(np.min(z), np.max(z))
    # exit()

    #y = 2 + 3 * x + 4 * x**2
    X = np.array([x, y]).T
    # X = features_polynomial_2d(x, y, degree=10)

    # X_train, X_test, target_train, target_test = train_test_split(X, z, test_size=0.5)
    # X_train = scaling(X_train)
    # target_train = scaling(target_train)

    # nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=50, L2=0.001,
    #                             output_activation_function=identity, hidden_activation_function=sigmoid, hidden_activation_derivative=sigmoid_derivative)
    nn = FeedForwardNeuralNetwork(X, z, n_hidden_layers=2, n_hidden_neurons=50, L2=0,
                                output_activation_function=identity, hidden_activation_function=sigmoid, hidden_activation_derivative=sigmoid_derivative)
    nn.train(SGD_const, evaluation_func=[MSE, R2], n_epochs=300, batch_size=32, init_lr=0.1, history=True)
    
    plt.plot(np.arange(300), nn.history[0])
    plt.plot(np.arange(300), nn.history[1])
    plt.show()

def nn_classification():
    # Import data

    data = pd.read_csv('data.csv')
    """
    The data file contains the following columns: 
    ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
    'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32']

    The column 'Unnamed: 32' only contains NaN values. 
    The id should not be relevant for the prediction. 
    I therefore drop these columns.
    The diagnosis corresponds to the target values.
    """

    diagnosis = data['diagnosis']
    diagnosis_int = (diagnosis == 'M')*1
    predictors = data.drop(['id','diagnosis','Unnamed: 32'], axis='columns')

    X = np.array(predictors)
    target = np.array(diagnosis_int)

    #Shuffle and split into training and test data
    from sklearn.model_selection import train_test_split
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.5)

    # Explore parameter space
    etas = np.logspace(-5,1,7)
    lmbds = np.logspace(-5,1,7)
    train_accuracy = np.zeros((len(etas), len(lmbds)))
    test_accuracy = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):

            instance = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=100, 
                                output_activation_function=sigmoid, hidden_activation_function=sigmoid, hidden_activation_derivative=sigmoid_derivative,
                                eta=eta, lmbd=lmbd, batch_size=100, n_epochs=50)
            instance.train_network()

            train_accuracy[i, j] = accuracy_score(target_train, instance.predict(X_train)) 
            test_accuracy[i, j] = accuracy_score(target_test, instance.predict(X_test))

    train_accuracy = pd.DataFrame(train_accuracy, columns = lmbds, index = etas)
    test_accuracy = pd.DataFrame(test_accuracy, columns = lmbds, index = etas)

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('figures/nn_classification/train_accuracy_ReLU.pdf')
    plt.savefig('figures/nn_classification/train_accuracy_ReLU.png')

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('figures/nn_classification/test_accuracy_ReLU.pdf')
    plt.savefig('figures/nn_classification/test_accuracy_ReLU.png')

def logistic_regression():

    data = pd.read_csv('data.csv')
    diagnosis = data['diagnosis']
    diagnosis_int = (diagnosis == 'M')*1
    predictors = data.drop(['id','diagnosis','Unnamed: 32'], axis='columns')

    X = np.array(predictors)
    target = np.array(diagnosis_int)


    #Shuffle and split into training and test data
    from sklearn.model_selection import train_test_split
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2)

    instance = LogisticRegression(X_train, target_train,L2=0.001)
    instance.train(SGD_ADAM, n_epochs=100, init_lr=0.1)
    print(instance.weights)
    import matplotlib.pyplot as plt
    plt.plot(np.arange(instance.n_epochs), instance.history)
    plt.show()
    quit()


    # Explore parameter space
    etas = np.logspace(-5,1,7)
    lmbds = np.logspace(-5,1,7)
    train_accuracy = np.zeros((len(etas), len(lmbds)))
    test_accuracy = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):

            instance = LogisticRegression(X_train, target_train,
                                    L2 = lmbd)
            instance.train(optimizer=StochasticGradientDescent, init_lr=eta, batch_size=100, n_epochs=50)

            train_accuracy[i, j] = accuracy_score(target_train, instance.predict(X_train)) 
            test_accuracy[i, j] = accuracy_score(target_test, instance.predict(X_test))

    import seaborn as sns
    import matplotlib.pyplot as plt

    train_accuracy = pd.DataFrame(train_accuracy, columns = lmbds, index = etas)
    test_accuracy = pd.DataFrame(test_accuracy, columns = lmbds, index = etas)

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('figures/lnreg_classification/train_accuracy.pdf')
    plt.savefig('figures/lnreg_classification/train_accuracy.png')

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('figures/lnreg_classification/test_accuracy.pdf')
    plt.savefig('figures/lnreg_classification/test_accuracy.png')


if __name__ == '__main__':
    # Set random seed
    random_seed=2023

    # Analysis of the GD and SGD codes
    # simple_polynomial_GD(random_seed)

    # Neural network regression
    nn_regression(random_seed)
