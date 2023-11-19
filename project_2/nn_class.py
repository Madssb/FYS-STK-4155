import numpy as np
from random import random, seed
from activation_functions import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative, identity
from SGD import *
np.random.seed(2023)

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

def MSE(target, prediction):
    return np.mean((target - prediction)**2)

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
        Initialize hidden weights and output weights using
        a normal distribution with mean=0 and standard deviation=1
        """
        hidden_weights = self.rng.normal(0, 1, (self.n_features, self.n_hidden_neurons))
        output_weights = self.rng.normal(0, 1, self.n_hidden_neurons)

        """
        Initialize hidden bias and output bias using a small number (0.01)
        """
        hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01
        output_bias = 0.01

        self.hidden_weights = hidden_weights
        self.output_weights = output_weights
        self.hidden_bias = hidden_bias
        self.output_bias = output_bias

        if self.n_hidden_layers > 1: 
            HIDDEN_weights = self.rng.normal(0, 1, (self.n_hidden_neurons, self.n_hidden_neurons, self.n_hidden_layers-1))
            HIDDEN_bias = np.zeros((self.n_hidden_neurons, self.n_hidden_layers-1)) + 0.01
            self.HIDDEN_weights = HIDDEN_weights
            self.HIDDEN_bias = HIDDEN_bias

    def feed_forward_pass(self, X):
        batch_size, n_features = np.shape(X)
        self.z_hidden = np.zeros((batch_size, self.n_hidden_neurons, self.n_hidden_layers))
        self.a_hidden = np.zeros((batch_size, self. n_hidden_neurons, self.n_hidden_layers))

        self.z_hidden[:,:,0] = X @ self.hidden_weights + self.hidden_bias
        self.a_hidden[:,:,0] = self.hidden_activation_function(self.z_hidden[:,:,0])
        if self.n_hidden_layers > 1: 
            for l in range(1,self.n_hidden_layers):
                # Since HIDDEN_weights and HIDDEN_bias apply to layers 2 and on, I have to index them
                # with "l-1" to get the l-th layer. 
                self.z_hidden[:,:,l] = self.a_hidden[:,:,l-1] @ self.HIDDEN_weights[:,:,l-1] + self.HIDDEN_bias[:,l-1]
                self.a_hidden[:,:,l] = self.hidden_activation_function(self.z_hidden[:,:,l])
        
        self.z_output = self.a_hidden[:,:,-1] @ self.output_weights + self.output_bias
        self.a_output = self.output_activation_function(self.z_output)

    def feed_forward_pass_out(self, X):
        n_inputs, n_features = np.shape(X)
        z_hidden = np.zeros((n_inputs, self.n_hidden_neurons, self.n_hidden_layers))
        a_hidden = np.zeros((n_inputs, self.n_hidden_neurons, self.n_hidden_layers))
        z_hidden[:,:,0] = X @ self.hidden_weights + self.hidden_bias
        a_hidden[:,:,0] = self.hidden_activation_function(z_hidden[:,:,0])
        if self.n_hidden_layers > 1:
            for l in range(1,self.n_hidden_layers):
                # Since HIDDEN_weights and_bias apply to layers 2 and on, the index l corresponds to get the (l+1)-th layer. 
                z_hidden[:,:,l] = a_hidden[:,:,l-1] @ self.HIDDEN_weights[:,:,l-1] + self.HIDDEN_bias[:,l-1]
                a_hidden[:,:,l] = self.hidden_activation_function(z_hidden[:,:,l])
        z_output = a_hidden[:,:,-1] @ self.output_weights + self.output_bias
        a_output = self.output_activation_function(z_output)
    
        return a_output
    
    def predict(self, X):
        a_output = self.feed_forward_pass_out(X)
        if self.output_activation_function == identity:
            return a_output
        else:
            return hard_classifier(a_output)

    def back_propagation(self, X, Y, model_parameters):
        batch_size, n_features = np.shape(X)
        if self.n_hidden_layers > 1:
            [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias, self.HIDDEN_weights, self.HIDDEN_bias] = model_parameters
        else:
            [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias] = model_parameters
        
        self.feed_forward_pass(X)
        
        error_output = self.a_output - Y
        self.output_weights_gradient = self.a_hidden[:,:,-1].T @ error_output
        self.output_bias_gradient = np.sum(error_output)
        
        error_output = np.expand_dims(error_output,1) # Broadcast the vector to allow matrix multiplication
        output_weights = np.expand_dims(self.output_weights,1) # Broadcast the vector to allow matrix multiplication

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
        
        self.hidden_weights_gradient = X.T @ error_hidden[:,:,0]
        self.hidden_bias_gradient = np.sum(error_hidden[:,:,0], axis=0)
        
        # Regularization term
        self.output_weights_gradient += self.L2 * self.output_weights
        self.hidden_weights_gradient += self.L2 * self.hidden_weights
        if self.n_hidden_layers > 1:
            self.HIDDEN_weights_gradient += self.L2*self.HIDDEN_weights_gradient

        if self.n_hidden_layers > 1:
            return [self.output_weights_gradient, self.output_bias_gradient, self.hidden_weights_gradient, self.hidden_bias_gradient, self.HIDDEN_weights_gradient, self.HIDDEN_bias_gradient]
        else:
            return [self.output_weights_gradient, self.output_bias_gradient, self.hidden_weights_gradient, self.hidden_bias_gradient]
    

    def train(self, optimizer, evaluation_func: callable, n_epochs=10, init_lr=0.1, batch_size=100, momentum=0.0):
        
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

        self.history = []

        for i in range(self.n_epochs):
            model_parameters = optimizer.advance(model_parameters)
            if self.n_hidden_layers > 1:
                [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias, self.HIDDEN_weights, self.HIDDEN_bias] = model_parameters
            else:
                [self.output_weights, self.output_bias, self.hidden_weights, self.hidden_bias] = model_parameters
            performance = evaluation_func(self.Y_full, self.predict(self.X_full))
            self.history.append(performance)


if __name__ == '__main__':
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

    #Shuffle and split into training and test data
    from sklearn.model_selection import train_test_split
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.5)

    instance = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=50, L2=0.001,
                                output_activation_function=sigmoid, hidden_activation_function=sigmoid, hidden_activation_derivative=sigmoid_derivative)
    instance.train(SGD_AdaGrad, evaluation_func=accuracy_score, n_epochs=50, batch_size=100, init_lr=0.01)
    import matplotlib.pyplot as plt
    plt.plot(np.arange(instance.n_epochs), instance.history)
    plt.show()


    # Explore parameter space
    etas = np.logspace(-5,1,7)
    lmbds = np.logspace(-5,1,7)
    train_accuracy = np.zeros((len(etas), len(lmbds)))
    test_accuracy = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):

            instance = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=100, L2=lmbd,
                                output_activation_function=sigmoid, hidden_activation_function=sigmoid, hidden_activation_derivative=sigmoid_derivative)
            instance.train(SGD_ADAM, evaluation_func=accuracy_score, n_epochs=50, batch_size=100, init_lr=eta)

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
    plt.savefig('figures/nn_classification/train_accuracy_class.pdf')
    plt.savefig('figures/nn_classification/train_accuracy_class.png')

    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('figures/nn_classification/test_accuracy_class.pdf')
    plt.savefig('figures/nn_classification/test_accuracy_class.png')