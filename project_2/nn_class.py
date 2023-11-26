# Parts of this code is inspired by Morten Hjort Jensen
# (https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/intro.html)

import numpy as np
from random import random, seed
from activation_functions import identity
np.random.seed(2023)

# Import SGD optimizers

from SGD import SGD_const, SGD_AdaGrad, SGD_RMSProp, SGD_ADAM

# Define activation functions

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return np.heaviside(x, np.zeros_like(x))

def leaky_ReLU(x):
    return np.maximum(0.01*x, x)

def leaky_ReLU_derivative(x, alpha=0.01):
    return np.where(x>0, 1, alpha)

def identity(x):
    return x

def identity_derivative(x):
    return 1

# Define validation metrics for validating performance

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

def MSE(target, prediction):
    return np.mean((target - prediction)**2)

def R2(target, prediction):
  SSE = np.sum((target - prediction)**2)
  Var = np.sum((target - np.mean(target))**2)
  return 1 - SSE/Var

class FeedForwardNeuralNetwork:

    """
    Feed Forward Neural Network
    """

    def __init__(self, X_data, Y_data,
               n_hidden_layers: int,
               n_hidden_neurons: int,
               output_activation_function: callable, 
               hidden_activation_function: object,
               hidden_activation_derivative: callable,
               L2 = 0.0, random_state=2023):
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
        L2 : float
        L2 regularization parameter (expected to be a float).
        random_state : int
        Given random state (expected to be an integer)

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
      

        # Initialize random state
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Initialize instance variables
        self.X_full = X_data
        self.Y_full = Y_data
        self.n_inputs, self.n_features = np.shape(X_data)
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.output_activation_function = output_activation_function
        self.hidden_activation_function = hidden_activation_function
        self.hidden_activation_derivative = hidden_activation_derivative
        self.L2 = L2

        # Initialize instance weights and biases
        self.initialize_weights_and_biases()
    
    def initialize_weights_and_biases(self):
        
        """
        Initialize hidden weights for the first layer and output weights using
        a normal distribution with mean=0 and standard deviation=1
        """
        hidden_weights = self.rng.normal(0, 1, (self.n_features, self.n_hidden_neurons))
        output_weights = self.rng.normal(0, 1, self.n_hidden_neurons)

        """
        Initialize hidden bias for the first layer and output bias using a small number (0.01)
        """
        hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01
        output_bias = 0.01

        self.hidden_weights = hidden_weights
        self.output_weights = output_weights
        self.hidden_bias = hidden_bias
        self.output_bias = output_bias

        """
        Initialize other hidden weights and bias in the case of multiple hidden layers
        """

        if self.n_hidden_layers > 1: 
            HIDDEN_weights = self.rng.normal(0, 1, (self.n_hidden_neurons, self.n_hidden_neurons, self.n_hidden_layers-1))
            HIDDEN_bias = np.zeros((self.n_hidden_neurons, self.n_hidden_layers-1)) + 0.01
            self.HIDDEN_weights = HIDDEN_weights
            self.HIDDEN_bias = HIDDEN_bias

    def feed_forward_pass(self, X):
        """
        Computes the feed-forward pass internally for the network. To be used in when training the model
        """

        # Create empty arrays for weighted sums and activated neurons for each datapoint in given batch
        batch_size, n_features = np.shape(X)
        self.z_hidden = np.zeros((batch_size, self.n_hidden_neurons, self.n_hidden_layers))
        self.a_hidden = np.zeros((batch_size, self. n_hidden_neurons, self.n_hidden_layers))

        # Compute weighted sum and activated values for first hidden layer
        self.z_hidden[:,:,0] = X @ self.hidden_weights + self.hidden_bias
        self.a_hidden[:,:,0] = self.hidden_activation_function(self.z_hidden[:,:,0])
        
        # Compute weighted sum and activated values for any other hidden layers
        if self.n_hidden_layers > 1: 
            for l in range(1,self.n_hidden_layers):
                # Since HIDDEN_weights and HIDDEN_bias apply to layers 2 and on, we index them
                # with "l-1" to get the l-th layer. 
                self.z_hidden[:,:,l] = self.a_hidden[:,:,l-1] @ self.HIDDEN_weights[:,:,l-1] + self.HIDDEN_bias[:,l-1]
                self.a_hidden[:,:,l] = self.hidden_activation_function(self.z_hidden[:,:,l])
        
        # Compute output weighted sum and final output
        self.z_output = self.a_hidden[:,:,-1] @ self.output_weights + self.output_bias
        self.a_output = self.output_activation_function(self.z_output)

    def feed_forward_pass_out(self, X):
        """
        Computed a feed-forward pass when given a new datasets. Returns the output, which is used for prediction
        """
        # Create empty arrays for weighted sums and activated neurons for each datapoint in input data
        n_inputs, n_features = np.shape(X)
        z_hidden = np.zeros((n_inputs, self.n_hidden_neurons, self.n_hidden_layers))
        a_hidden = np.zeros((n_inputs, self.n_hidden_neurons, self.n_hidden_layers))
        
        # Compute weighted sum and activated values for first hidden layer
        z_hidden[:,:,0] = X @ self.hidden_weights + self.hidden_bias
        a_hidden[:,:,0] = self.hidden_activation_function(z_hidden[:,:,0])
        
        # Compute weighted sum and activated values for any other hidden layers
        if self.n_hidden_layers > 1:
            for l in range(1,self.n_hidden_layers):
                # Since HIDDEN_weights and_bias apply to layers 2 and on, the index l corresponds to get the (l+1)-th layer. 
                z_hidden[:,:,l] = a_hidden[:,:,l-1] @ self.HIDDEN_weights[:,:,l-1] + self.HIDDEN_bias[:,l-1]
                a_hidden[:,:,l] = self.hidden_activation_function(z_hidden[:,:,l])
        
        # Compute output weighted sum and final output
        z_output = a_hidden[:,:,-1] @ self.output_weights + self.output_bias
        a_output = self.output_activation_function(z_output)
    
        return a_output
    
    def predict(self, X):
        """
        Gives a prediction dependent on the output activation function. Performs a hard classification if the output is probability
        """
        a_output = self.feed_forward_pass_out(X)
        if self.output_activation_function==identity:
            return a_output
        else:
            return hard_classifier(a_output)

    def back_propagation(self, X, Y, model_parameters):
        """
        The backpropagation algorithm for training the neural network. Outputs the gradients of the weights and biases
        """
        batch_size, n_features = np.shape(X)
        if self.n_hidden_layers > 1:
            [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias, self.HIDDEN_weights, self.HIDDEN_bias] = model_parameters
        else:
            [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias] = model_parameters
        
        # Perform feed-forward pass
        self.feed_forward_pass(X)
        
        # Evaluate cost of the output and compute gradients for output weights and bias
        error_output = self.a_output - Y
        self.output_weights_gradient = self.a_hidden[:,:,-1].T @ error_output
        self.output_bias_gradient = np.sum(error_output)
        
        # Broadcast vectors to allow matrix multiplication
        error_output = np.expand_dims(error_output,1) 
        output_weights = np.expand_dims(self.output_weights,1)

        # Compute gradients by backpropagating cost through hidden layers
        error_hidden = np.zeros((batch_size, self.n_hidden_neurons, self.n_hidden_layers))
        error_hidden[:,:,-1] = error_output @ output_weights.T * self.hidden_activation_derivative(self.z_hidden[:,:,-1])

        if self.n_hidden_layers > 1: 
            self.HIDDEN_weights_gradient = np.zeros((self.n_hidden_neurons, self.n_hidden_neurons, self.n_hidden_layers-1))
            self.HIDDEN_bias_gradient = np.zeros((self.n_hidden_neurons, self.n_hidden_layers-1))
            self.HIDDEN_weights_gradient[:,:,-1] = self.a_hidden[:,:,-2].T @ error_hidden[:,:,-1]
            self.HIDDEN_bias_gradient[:,-1] = np.sum(error_hidden[:,:,-1], axis=0)
            for l in range(self.n_hidden_layers-2, 0,-1):
                # Since HIDDEN_weights and_bias apply to layers 2 and on, the index l corresponds to get the (l+1)-th layer. 
                error_hidden[:,:,l] = error_hidden[:,:,l+1] @ self.HIDDEN_weights[:,:,l].T * self.hidden_activation_derivative(self.z_hidden[:,:,l])
                self.HIDDEN_weights_gradient[:,:,l-1] = self.a_hidden[:,:,l-1].T @ error_hidden[:,:,l]
                self.HIDDEN_bias_gradient[:,l-1] = np.sum(error_hidden[:,:,l], axis=0)
            error_hidden[:,:,0] = error_hidden[:,:,1] @ self.HIDDEN_weights[:,:,0].T * self.hidden_activation_derivative(self.z_hidden[:,:,0])
        
        # Gradients in the first hidden layer
        self.hidden_weights_gradient = X.T @ error_hidden[:,:,0]
        self.hidden_bias_gradient = np.sum(error_hidden[:,:,0], axis=0)
        
        # Add regularization term from cost function to gradients
        self.output_weights_gradient += self.L2 * self.output_weights
        self.hidden_weights_gradient += self.L2 * self.hidden_weights
        if self.n_hidden_layers > 1:
            self.HIDDEN_weights_gradient += self.L2*self.HIDDEN_weights_gradient

        # Output gradients
        if self.n_hidden_layers > 1:
            return [self.output_weights_gradient, self.output_bias_gradient, self.hidden_weights_gradient, self.hidden_bias_gradient, self.HIDDEN_weights_gradient, self.HIDDEN_bias_gradient]
        else:
            return [self.output_weights_gradient, self.output_bias_gradient, self.hidden_weights_gradient, self.hidden_bias_gradient]
    

    def train(self, optimizer, n_epochs=10, init_lr=0.1, batch_size=100, momentum=0.0, evaluation_func=None, history=False):
        """
        Train the model using an SGD optimizer of choice
        """
        self.n_epochs = n_epochs
        if self.n_hidden_layers > 1:
            model_parameters = [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias, self.HIDDEN_weights, self.HIDDEN_bias]
        else:
            model_parameters = [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias]
        
        # Build optimizer
        optimizer = optimizer(self.X_full, self.Y_full, 
                              gradient_func=self.back_propagation, 
                              init_model_parameters=model_parameters,
                              init_lr = init_lr, batch_size=batch_size,
                              momentum = momentum,random_state=self.random_state)
        
        # Empty array to store history of model performance
        if history == True:
            self.history = np.zeros((len(evaluation_func) ,n_epochs))


        for i in range(self.n_epochs):
            # Perform gradient descent-based optimization
            model_parameters = optimizer.advance(model_parameters)
            # Update model parameters
            if self.n_hidden_layers > 1:
                [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias, self.HIDDEN_weights, self.HIDDEN_bias] = model_parameters
            else:
                [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias] = model_parameters
            
            # Store model performance
            if history == True:
                for j, eval_func in enumerate(evaluation_func):
                    self.history[j, i] = eval_func(t_test, self.predict(X_test))