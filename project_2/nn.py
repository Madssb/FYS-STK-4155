import numpy as np
from random import random, seed
from activation_functions import sigmoid
np.random.seed(2023)

# Activation function
# def sigmoid(x):
#     return 1./(1 + np.exp(-x))

def sigmoid_derivative(f: float):
    """
    Given a sigmoid function f(x), this will be its derivative df/dx
    """
    return f * (1 - f)

# Cross-entropy cost function
def cross_entropy_binary(target, probability):
    sum = 0
    for i in range(len(target)):
        sum += target[i]*np.log(probability[i]) + (1 - target[i])*np.log(1 - probability[i])
    return -sum

# Accuracy score functions

def indicator(target, probability):
    if probability >= 0.5:
        # If the probability of predicting 1 is 0.5 or higher, I classify this as a prediction of 1
        # I then want the indicator function to signal if the prediction is correct
        # I do this by returning the target, which will give 1 if the target is also 1, 0 if it is 0
        return target 
    if probability < 0.5 and target == 0:
        # If the probability of predicting 1 is lower than 0.5, I classify this as a prediction of 0
        # If the target is also 0 I, I signal that the prediction is correct by return 1
        return 1
    else:
        # If the target is not 0, I signal that the prediction is not correct by returning 0
        return 0
    
def accuracy_score(target, probability):
    """
    Returns the average number of correct predictions
    """
    n = len(target)
    assert len(probability) == n, "Not the same number of predictions as targets"
    sum = 0
    for i in range(n):
        sum += indicator(target[i], probability[i])
    return sum/n

# Feed-forward pass algorithm

def feed_forward_pass(X, hidden_weights, output_weights, hidden_bias, output_bias):
    z_hidden = X @ hidden_weights + hidden_bias
    a_hidden = sigmoid(z_hidden)
    z_output = a_hidden @ output_weights + output_bias
    a_output = sigmoid(z_output)
    
    return a_hidden, a_output

# Backpropagation algorithm

def back_propagation(X, Y, hidden_weights, output_weights, hidden_bias, output_bias):
    
    a_hidden, a_output = feed_forward_pass(X, hidden_weights, output_weights, hidden_bias, output_bias)
    
    error_output = a_output - Y
    output_weights_gradient = a_hidden.T @ error_output
    output_bias_gradient = np.sum(error_output)
    error_output = np.expand_dims(error_output,1) # Broadcast the vector to allow matrix multiplication
    output_weights = np.expand_dims(output_weights,1) # Broadcast the vector to allow matrix multiplication
    
    error_hidden = error_output @ output_weights.T * sigmoid_derivative(a_hidden)
    hidden_weights_gradient = X.T @ error_hidden
    hidden_bias_gradient = np.sum(error_hidden)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient, a_hidden, a_output

def train_model(X, target, n_hidden_neurons, eta, lmbd, momentum, n_iterations):
    
    np.random.seed(2023)
    
    # Initialize weights and biases
    n_inputs, n_features = np.shape(X)

    hidden_weights = np.random.normal(0, 1, (n_features, n_hidden_neurons))
    hidden_bias = np.zeros(n_hidden_neurons) + 0.01

    output_weights = np.random.normal(0, 1, n_hidden_neurons)
    output_bias = 0.01
    
    # Perform feed-forward pass
    a_hidden, a_output = feed_forward_pass(X, 
                                           hidden_weights, output_weights, 
                                           hidden_bias, output_bias)
    #print("Pre-training probability: ", a_output)
    
        
    change_Wo = 0
    change_bo = 0
    change_Wh = 0
    change_bh = 0
    for i in range(n_iterations):
        # calculate gradients
        dWo, dBo, dWh, dBh, a_hidden, a_output = back_propagation(X, target, 
                                                                  hidden_weights, output_weights, 
                                                                  hidden_bias, output_bias)
        
        # regularization term gradients
        dWo += lmbd * output_weights
        dWh += lmbd * hidden_weights
        
        # update weights and biases
        # here I use a fixed learning rate with the possibility to apply momentum
        output_weights -= eta * dWo + change_Wo*momentum
        output_bias -= eta * dBo + change_bo*momentum
        hidden_weights -= eta * dWh + change_Wh*momentum
        hidden_bias -= eta * dBh + change_bh*momentum
        change_Wo = eta * dWo
        change_bo = eta * dBo
        change_Wh = eta * dWh
        change_bh = eta * dBh
    
    #print("Accuracy score: ",accuracy_score(target, a_output))
    #print("Final probability: ", a_output)
    #print("Target: ", target)
    #print("Differences: ", target - a_output)

    return accuracy_score(target, a_output)



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

# Shuffle and split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2)

# Explore parameter space
etas = np.logspace(-5,1,7)
lmbds = np.logspace(-5,1,7)
train_accuracy = np.zeros((len(etas), len(lmbds)))

for i, eta in enumerate(etas):
    for j, lmbd in enumerate(lmbds):
        
        train_accuracy[i, j] = train_model(X_train, target_train, 30, eta=eta, lmbd=lmbd, momentum=0.0, n_iterations=1000)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
