
# Import external packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import warnings
import seaborn as sns

# Import self-made packages
from nn_class import FeedForwardNeuralNetwork
from nn_class import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative, identity
from nn_class import hard_classifier, indicator, accuracy_score, MSE
from SGD import SGD_const, SGD_AdaGrad, SGD_RMSProp, SGD_ADAM

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

x = np.expand_dims(np.linspace(-10, 10, 100), 1)
y = 2 + 3 * x + 4 * x**2
X = np.array([x, x]).T

nn = FeedForwardNeuralNetwork(x, y, n_hidden_layers=2, n_hidden_neurons=20, L2=0.001,
                            output_activation_function=identity, hidden_activation_function=leaky_ReLU, hidden_activation_derivative=leaky_ReLU_derivative)
nn.train(SGD_ADAM, evaluation_func=MSE, n_epochs=100, batch_size=10, init_lr=0.1)

plt.plot(x, y, label='real')
plt.plot(x, nn.predict(x), label='model')
plt.legend()
plt.show()

quit()
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
