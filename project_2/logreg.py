import numpy as np
from activation_functions import sigmoid
np.random.seed(2023)

class LogisticRegression:

    def __init__(self, X_data, Y_data,
                  eta=0.1, lmbd=0.01, batch_size=100, n_epochs=50,
                  random_state=2023):
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
        self.eta = eta
        self.lmbd = lmbd
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_iterations = self.n_inputs // batch_size       

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

    
    def feed_forward(self):

        self.z = self.X @ self.weights + self.bias
        self.probabilities = self.activation(self.z)

    def predict(self, X):
        z = X @ self.weights + self.bias
        probabilities = self.activation(z)
        return hard_classifier(probabilities)
    
    def back_propagation(self):
    
        self.feed_forward()
        
        # Derivative of cross-entropy loss
        error = self.probabilities - self.Y
        self.weights_gradient = self.X.T @ error
        self.bias_gradient = np.sum(error)

    def train(self):
        # Fix random seed
        np.random.seed(2023)
        
        data_indices = np.arange(self.n_inputs)
        self.history = []
        for i in range(self.n_epochs):
            for j in range(self.n_iterations):
                # pick datapoints with replacement
                batch_datapoints = self.rng.choice(data_indices, size=self.batch_size, replace=False)
                
                # set up minibatch with training data
                self.X = self.X_full[batch_datapoints]
                self.Y = self.Y_full[batch_datapoints]

                # update gradients
                self.back_propagation()
            
                # regularization term gradients
                self.weights_gradient += 2 * self.lmbd * self.weights

                # update weights and biases
                # here I use a fixed learning rate with the possibility to apply momentum
                self.weights -= self.eta * self.weights_gradient 
                self.bias -= self.eta * self.bias_gradient 

            performance = accuracy_score(self.Y_full,self.predict(self.X_full))
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

instance = LogisticRegression(X_train, target_train,
                            eta=0.0001, lmbd=0.01, batch_size=100, n_epochs=100)
instance.train()
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
                            eta=eta, lmbd=lmbd, batch_size=100, n_epochs=50)
        instance.train()

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
