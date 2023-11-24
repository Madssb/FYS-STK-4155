"""
Apply sklearn to problems we solve with own code for comparison sake.
"""
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def regression_1d_poly_2_deg():
  """Scikit-Learn OLS optimization
  """
# Set a random seed for reproducibility
  np.random.seed(2024)
  rng = np.random.default_rng(2023)
  n = 100
  x = np.linspace(-10, 10, n)
  y = 4 + 3 * x + x**2 + rng.normal(0, 0.1, n)

  X = np.array([np.ones(100), x, x**2]).T
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split the data into train and test sets

  # Create and train the MLPRegressor
  model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)
  model.fit(X_train, y_train)

  # Make predictions on the test set
  predicted = model.predict(X_test)

  # Calculate mean squared error
  mse = mean_squared_error(y_test, predicted)
  print("Mean Squared Error:", mse)


regression_1d_poly_2_deg()


def classification_logistic():
    """Scikit-Learn training of neural network with sigmoid activation funcs
    """
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
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.5)

    # Explore parameter space
    etas = np.logspace(-5,1,7)
    lmbds = np.logspace(-5,1,7)
    train_accuracy = np.zeros((len(etas), len(lmbds)))
    test_accuracy = np.zeros((len(etas), len(lmbds)))
    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
          instance = MLPClassifier(hidden_layer_sizes=(100,100),learning_rate_init=eta, alpha=lmbd, activation='logistic')
          instance.fit(X_train, target_train)
          predict = instance.predict(X_test)
          trained = instance.predict(X_train)
          train_accuracy[i, j] =  mean_squared_error(target_train, trained)
          test_accuracy[i, j] = mean_squared_error(target_test, predict)

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('figures/nn_classification/train_accuracy_ReLu_scikit.pdf')
    plt.savefig('figures/nn_classification/train_accuracy_ReLu_scikit.png')

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('figures/nn_classification/test_accuracy_sigmoid_scikit.pdf')
    plt.savefig('figures/nn_classification/test_accuracy_sigmoid_scikit.png')

def classification_relu():
    """Scikit-Learn training of neural network with ReLu activation funcs
    """
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
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.5)

    # Explore parameter space
    etas = np.logspace(-5,1,7)
    lmbds = np.logspace(-5,1,7)
    train_accuracy = np.zeros((len(etas), len(lmbds)))
    test_accuracy = np.zeros((len(etas), len(lmbds)))
    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
          instance = MLPClassifier(hidden_layer_sizes=(100,100),learning_rate_init=eta, alpha=lmbd, activation='relu')
          instance.fit(X_train, target_train)
          predict = instance.predict(X_test)
          trained = instance.predict(X_train)
          train_accuracy[i, j] =  mean_squared_error(target_train, trained)
          test_accuracy[i, j] = mean_squared_error(target_test, predict)

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('figures/nn_classification/train_accuracy_ReLu_scikit.pdf')
    plt.savefig('figures/nn_classification/train_accuracy_ReLu_scikit.png')

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('figures/nn_classification/test_accuracy_ReLu_scikit.pdf')
    plt.savefig('figures/nn_classification/test_accuracy_ReLu_scikit.png')


classification_logistic()
classification_relu()


