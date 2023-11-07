import numpy as np
from random import random, seed
np.random.seed(2023)

# Activation function
def sigmoid(x):
    return 1./(1 + np.exp(-x))

def sigmoid_derivative(f):
    """
    Given a sigmoid function f(x), this will be its derivative df/dx
    """
    return f * (1 - f)

    
# Accuracy score functions for classification

def hard_classifier(probability):
    return (probability >= 0.5)*1

def indicator(target, prediction):
    return(target==prediction)*1
    
def accuracy_score(target, prediction):
    """
    Returns the average number of correct predictions
    """
    n = len(target)
    assert len(prediction) == n, "Not the same number of predictions as targets"
    return np.sum(indicator(target, prediction))/n


class FeedForwardNeuralNetwork:

    """
    Feed Forward Neural Network
    """

    def __init__(self, X, Y, 
               n_hidden_layers: int,
               n_hidden_neurons: int,
               output_activation_function: callable, 
               hidden_activation_function: callable,
               hidden_activation_derivative: callable,
               eta=0.1, lmbd=0.01, momentum=0, n_iterations=10):
        """Constructor

        Parameters
        ----------
        n_inputs : int
        Number of inputs (expected to be an integer).
        n_hidden_layers : int
        Number of hidden layers (expected to be an integer).
        n_neurons : int
        Number of neurons per hidden layer (expected to be an integer).
        n_outputs : int
        Number of nodes in output (expected to be an integer).
        activation_function : callable
        Activation function (expected to be a callable function).

        Raises
        ------
        TypeError
            If any of the input parameters does not have the expected type.
        """
        # Check types of input parameters
        if not isinstance(n_hidden_layers, int):
            raise TypeError("n_hidden_layers must be an integer.")
        if not isinstance(n_hidden_neurons, int):
            raise TypeError("n_neurons must be an integer.")
        if not callable(output_activation_function):
            raise TypeError("output_activation_function must be a callable function.")
        if not callable(hidden_activation_function):
            raise TypeError("hidden_activation_function must be a callable function.")
        if not callable(hidden_activation_derivative):
            raise TypeError("hidden_activation_derivative must be a callable function.")
        # Initialize instance variables
        self.X = X
        self.target = Y
        self.n_inputs, self.n_features = np.shape(X)
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.output_activation_function = output_activation_function
        self.hidden_activation_function = hidden_activation_function
        self.hidden_activation_derivative = hidden_activation_derivative

        # Initialize instance weights and biases
        self.initialize_weights_and_biases()

        # Initialize optimization parameters
        self.eta = eta
        self.lmbd = lmbd
        self.momentum = momentum
        self.n_iterations = n_iterations
    
    def initialize_weights_and_biases(self):
        
        # Fix random seed
        np.random.seed(2023)
        
        """
        Initialize hidden weights and output weights using
        a normal distribution with mean=0 and standard deviation=1
        """
        hidden_weights = np.random.normal(0, 1, (self.n_features, self.n_hidden_neurons))
        #HIDDEN_weights = np.random.normal(0, 1, (self.n_hidden_neurons, self.n_hidden_layers-1))
        output_weights = np.random.normal(0, 1, self.n_hidden_neurons)

        """
        Initialize hidden bias and output bias using a small number (0.01)
        """
        hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01
        #HIDDEN_bias = np.zeros(self.n_hidden_neurons, self.n_hidden_layers -1 ) + 0.01
        output_bias = 0.01

        self.hidden_weights = hidden_weights
        self.output_weights = output_weights
        self.hidden_bias = hidden_bias
        self.output_bias = output_bias
        #self.HIDDEN_weights = HIDDEN_weights
        #self.HIDDEN_bias = HIDDEN_bias

    def feed_forward_pass(self):
        self.z_hidden = self.X @ self.hidden_weights + self.hidden_bias
        self.a_hidden = self.hidden_activation_function(self.z_hidden)

        self.z_output = self.a_hidden @ self.output_weights + self.output_bias
        self.a_output = self.output_activation_function(self.z_output)
    
    def feed_forward_pass_out(self, X):
        z_hidden = X @ self.hidden_weights + self.hidden_bias
        a_hidden = self.hidden_activation_function(z_hidden)
        z_output = self.a_hidden @ self.output_weights + self.output_bias
        a_output = self.output_activation_function(z_output)
    
        return a_output
    
    def back_propagation(self):
    
        self.feed_forward_pass()
        
        error_output = self.a_output - self.target
        self.output_weights_gradient = self.a_hidden.T @ error_output
        self.output_bias_gradient = np.sum(error_output)
        
        error_output = np.expand_dims(error_output,1) # Broadcast the vector to allow matrix multiplication
        output_weights = np.expand_dims(self.output_weights,1) # Broadcast the vector to allow matrix multiplication
        
        error_hidden = error_output @ output_weights.T * self.hidden_activation_derivative(self.a_hidden)
        self.hidden_weights_gradient = self.X.T @ error_hidden
        self.hidden_bias_gradient = np.sum(error_hidden)

    def train_network(self):
        
        np.random.seed(2023)
            
        change_Wo = 0
        change_bo = 0
        change_Wh = 0
        change_bh = 0
        for i in range(self.n_iterations):
            # update gradients
            self.back_propagation()
            
            # regularization term gradients
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights
            
            # update weights and biases
            # here I use a fixed learning rate with the possibility to apply momentum
            self.output_weights -= self.eta * self.output_weights_gradient + change_Wo*self.momentum
            self.output_bias -= self.eta * self.output_bias_gradient + change_bo*self.momentum
            self.hidden_weights -= self.eta * self.hidden_weights_gradient + change_Wh*self.momentum
            self.hidden_bias -= self.eta * self.hidden_bias_gradient + change_bh*self.momentum
            change_Wo = self.eta * self.output_weights_gradient
            change_bo = self.eta * self.output_bias_gradient
            change_Wh = self.eta * self.hidden_weights_gradient
            change_bh = self.eta * self.hidden_bias_gradient

    def predict(self, X, problem='Classification'):
        a_output = self.feed_forward_pass_out(X)
        assert problem=='Classification' or problem=='Regression', "Must be a 'Classification' or 'Regression' problem"
        if problem=='Classification':
            return hard_classifier(a_output)
        else:
            return a_output


# Import data

import pandas as pd 

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

instance = FeedForwardNeuralNetwork(X, target, n_hidden_layers=1, n_hidden_neurons=3, 
                            output_activation_function=sigmoid, hidden_activation_function= sigmoid, hidden_activation_derivative=sigmoid_derivative,
                            eta=0.1, lmbd=0.01, momentum=0, n_iterations=100)
print(instance.n_features)
print(instance.n_inputs)

instance.train_network()
prediction=instance.predict(X, problem='Classification')
print(accuracy_score(target, prediction))

quit()
# Shuffle and split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2)

# Explore parameter space
etas = np.logspace(-5,1,7)
lmbds = np.logspace(-5,1,7)
train_accuracy = np.zeros((len(etas), len(lmbds)))

for i, eta in enumerate(etas):
    for j, lmbd in enumerate(lmbds):

        train_accuracy[i, j] = train_model(X_train, target_train, 200, eta=eta, lmbd=lmbd, momentum=0.0, n_iterations=1000)

import seaborn as sns
import matplotlib.pyplot as plt

train_accuracy = pd.DataFrame(train_accuracy, columns = lmbds, index = etas)

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
