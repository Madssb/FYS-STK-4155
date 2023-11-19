import numpy as np
np.random.seed(2023)
from activation_functions import sigmoid
from gradient_descents import StochasticGradientDescent, Adagrad, RMSProp, ADAM
from activation_functions import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative

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
    
    def train(self, optimizer, n_epochs=10, init_lr=0.1, batch_size=100, momentum=0.0):

        self.n_epochs = n_epochs
        model_parameters = [self.weights, self.bias]
        # Build optimizer
        optimizer = optimizer(self.X_full, self.Y_full, 
                              gradient_func=self.back_propagation, 
                              init_model_parameters=model_parameters,
                              init_lr = init_lr, batch_size=batch_size,
                              momentum = momentum,random_state=self.random_state)
        
        self.history = []

        for i in range(self.n_epochs):
            model_parameters = optimizer.advance(model_parameters)
            [self.weights, self.bias] = model_parameters
            performance = self.evaluation(self.Y_full,self.predict(self.X_full))
            self.history.append(performance)

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
X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2)

instance = LogisticRegression(X_train, target_train,L2=0.001)
instance.train(ADAM, n_epochs=100, init_lr=0.1)
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
