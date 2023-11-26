import numpy as np
np.random.seed(2023)

from activation_functions import sigmoid
from utilities import hard_classifier, indicator, accuracy_score

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

class LogisticRegression:

    def __init__(self, X_data, Y_data,
                  L2 = 0.0, random_state=2023):
        # Initialize random state
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Initialize instance variables
        self.X_full = X_data
        self.Y_full = Y_data
        self.X = X_data
        self.Y = Y_data

        self.n_inputs, self.n_features = np.shape(X_data)
        self.activation = sigmoid

        self.initialize_weights_and_biases()
 
        # Initialize optimization parameters
        self.evaluation = accuracy_score
        self.L2 = L2   

    def initialize_weights_and_biases(self):
        
        """
        Initialize output weights using
        a normal distribution with mean=0 and standard deviation=1
        """
        weights = self.rng.normal(0, 1, self.n_features)

        """
        Initialize output bias using a small number (0.01)
        """
        bias = 0.01

        self.weights = weights
        self.bias = bias   

    
    def feed_forward(self, X):

        self.z = X @ self.weights + self.bias
        self.probabilities = self.activation(self.z)

    def predict(self, X):
        z = X @ self.weights + self.bias
        probabilities = self.activation(z)
        return hard_classifier(probabilities)
    
    def back_propagation(self, X, Y, model_parameters):
        
        [self.weights, self.bias] = model_parameters
        self.feed_forward(X)
        
        # Derivative of cross-entropy loss
        error = self.probabilities - Y
        self.weights_gradient = X.T @ error
        self.bias_gradient = np.sum(error)

        # Regularization term
        self.weights_gradient += self.L2 * self.weights

        return [self.weights_gradient, self.bias_gradient]
    
    def train(self, optimizer, n_epochs=10, init_lr=0.1, batch_size=100, momentum=0.0, history=False, t_test=None, X_test=None):

        self.n_epochs = n_epochs
        model_parameters = [self.weights, self.bias]
        # Build optimizer
        optimizer = optimizer(self.X_full, self.Y_full, 
                              gradient_func=self.back_propagation, 
                              init_model_parameters=model_parameters,
                              init_lr = init_lr, batch_size=batch_size,
                              momentum = momentum, random_state=self.random_state)
        
        if history == True:
            self.history = np.zeros(n_epochs)

        for i in range(self.n_epochs):
            model_parameters = optimizer.advance(model_parameters)
            [self.weights, self.bias] = model_parameters
            if history == True:
                self.history[i] = self.evaluation(t_test, self.predict(X_test))
           