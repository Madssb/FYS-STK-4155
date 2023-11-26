""" File for generating Feedforward neural network figures and logistic regression figures. """

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import warnings
import seaborn as sns

from activation_functions import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative, identity
from nn_class import FeedForwardNeuralNetwork
from nn_class import accuracy_score, MSE, R2
from SGD import SGD_const, SGD_AdaGrad, SGD_RMSProp, SGD_ADAM
from logreg_gradient_class import LogisticRegression
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from utilities import franke_function
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning



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




def nn_regression_network_OLS_batchsize(learning_method=SGD_const):
    np.random.seed(2023)
    rng = np.random.RandomState(2023)
    n = 40
    points = n*n
    x = rng.rand(points)
    y = rng.rand(points)
    # x = np.arange(0, 1, 1/points)
    # y = np.arange(0, 1, 1/points)
    z = franke_function(x, y)
    noise = np.random.normal(0, 1, z.shape)*0.1
    z = z + noise
    X = np.array([x, y]).T

    X_train, X_test, target_train, target_test = train_test_split(X, z, test_size=0.2)

    etas = np.logspace(0, -5, 6)
    epochs = 300
    batch_sizes = [2**i for i in range(3, 8)]

    MSE_scores = np.zeros((len(batch_sizes),len(etas)))
    R2_scores = np.zeros((len(batch_sizes),len(etas)))

    # constant learning rate

    for i, batch_s in enumerate(batch_sizes):
        for j, eta in enumerate(etas):
            # nn = FeedForwardNeuralNetwork(X, z, n_hidden_layers=int(l), n_hidden_neurons=n, L2=0,
            #                               output_activation_function=identity,
            #                               hidden_activation_function=sigmoid,
            #                               hidden_activation_derivative=sigmoid_derivative)
            nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=0,
                                          output_activation_function=identity,
                                          hidden_activation_function=sigmoid,
                                          hidden_activation_derivative=sigmoid_derivative)
            nn.train(learning_method, n_epochs=epochs, init_lr=eta, batch_size=batch_s)
            MSE_scores[i, j] = MSE(target_test, nn.predict(X_test))
            R2_scores[i, j] = R2(target_test, nn.predict(X_test))

    MSE_scores[MSE_scores > 1] = np.nan
    R2_scores[R2_scores > 1] = np.nan
    R2_scores[-1 > R2_scores] = np.nan
    
    MSE_scores = pd.DataFrame(MSE_scores, columns = etas, index = batch_sizes)
    R2_scores = pd.DataFrame(R2_scores, columns = etas, index = batch_sizes)

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(MSE_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'})
    x_ticks = [fr'$10^{{{int(np.log10(eta))}}}$' for eta in etas]
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel("Learning rate $\eta$")
    ax.set_ylabel("Batch size")
    plt.tight_layout()
    plt.savefig(f'figures/nn_regression/batchsize_{learning_method.__name__}_OLS_MSE_nn_regression__epochs_{epochs}__Franke_{n}_points.pdf', bbox_inches='tight')

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(R2_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': r'$R^2$ score'})
    x_ticks = [fr'$10^{{{int(np.log10(eta))}}}$' for eta in etas]
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel("Learning rate $\eta$")
    ax.set_ylabel("Batch size")
    plt.savefig(f'figures/nn_regression/batchsize_{learning_method.__name__}_OLS_R2_nn_regression__epochs_{epochs}__Franke_{n}_points.pdf', bbox_inches='tight')


def nn_regression_network_OLS_structure(learning_method=SGD_const):
    rng = np.random.RandomState(2023)
    n = 40
    points = n*n
    x = rng.rand(points)
    y = rng.rand(points)
    # x = np.arange(0, 1, 1/points)
    # y = np.arange(0, 1, 1/points)
    z = franke_function(x, y)
    noise = np.random.normal(0, 1, z.shape)*0.1
    z = z + noise
    X = np.array([x, y]).T

    X_train, X_test, target_train, target_test = train_test_split(X, z, test_size=0.2)

    neurons = [5, 10, 30, 50]
    layers = [1, 2, 3]
    eta = 0.01
    epochs = 300
    batch_size = 16

    MSE_lay_neur = np.zeros((len(layers),len(neurons)))
    R2_lay_neur = np.zeros((len(layers),len(neurons)))

    # constant learning rate

    for i, l in enumerate(layers):
        for j, n in enumerate(neurons):
            # nn = FeedForwardNeuralNetwork(X, z, n_hidden_layers=int(l), n_hidden_neurons=n, L2=0,
            #                               output_activation_function=identity,
            #                               hidden_activation_function=sigmoid,
            #                               hidden_activation_derivative=sigmoid_derivative)
            nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=int(l), n_hidden_neurons=n, L2=0,
                                          output_activation_function=identity,
                                          hidden_activation_function=sigmoid,
                                          hidden_activation_derivative=sigmoid_derivative)
            nn.train(learning_method, n_epochs=epochs, init_lr=eta, batch_size=batch_size)
            MSE_lay_neur[i, j] = MSE(target_test, nn.predict(X_test))
            R2_lay_neur[i, j] = R2(target_test, nn.predict(X_test))
    
    MSE_scores = pd.DataFrame(MSE_lay_neur, columns = neurons, index = layers)
    R2_scores = pd.DataFrame(R2_lay_neur, columns = neurons, index = layers)

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(MSE_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'})
    # ax.set_title(f'SGD_const MSE')
    ax.set_xlabel("Neurons")
    ax.set_ylabel("Layers")
    plt.tight_layout()
    plt.savefig(f'figures/nn_regression/structure_{learning_method.__name__}_OLS_MSE_nn_regression_eta_{eta}__epochs_{epochs}_batch_size_{batch_size}.pdf', bbox_inches='tight')

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(R2_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': r'$R^2$ score'})
    # ax.set_title(f'SGD_const R2')
    ax.set_xlabel("$Neurons$")
    ax.set_ylabel("$Layers$")
    plt.tight_layout()
    plt.savefig(f'figures/nn_regression/structure_{learning_method.__name__}_OLS_R2_nn_regression_eta_{eta}__epochs_{epochs}_batch_size_{batch_size}.pdf', bbox_inches='tight')

    # plt.show()

def nn_regression_network_eta_l2(learning_method=SGD_const, layer_func=sigmoid):
    np.random.seed(2023)
    rng = np.random.RandomState(2023)
    n = 40
    points = n*n
    x = rng.rand(points)
    y = rng.rand(points)
    z = franke_function(x, y)
    noise = np.random.normal(0, 1, z.shape)*0.1
    z = z + noise
    X = np.array([x, y]).T

    X_train, X_test, target_train, target_test = train_test_split(X, z, test_size=0.2)

    etas = np.logspace(0, -5, 6)
    n_l2 = 7
    l2s = np.zeros(n_l2)
    l2s[:-1] = np.logspace(0, -5, 6)
    epochs= 300
    batch_size = 16

    MSE_scores = np.zeros((len(l2s), len(etas)))
    R2_scores = np.zeros((len(l2s), len(etas)))


    for i, l2 in enumerate(l2s):
        for j, eta in enumerate(etas):

            if layer_func == sigmoid:
                nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2,
                                            output_activation_function=identity,
                                            hidden_activation_function=sigmoid,
                                            hidden_activation_derivative=sigmoid_derivative)
            elif layer_func == ReLU:
                nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2,
                                            output_activation_function=identity,
                                            hidden_activation_function=ReLU,
                                            hidden_activation_derivative=ReLU_derivative)
            elif layer_func == leaky_ReLU:
                nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2,
                                            output_activation_function=identity,
                                            hidden_activation_function=leaky_ReLU,
                                            hidden_activation_derivative=leaky_ReLU_derivative)
            nn.train(optimizer=learning_method, n_epochs=epochs, init_lr=eta, batch_size=batch_size)#len(z))
            MSE_scores[i, j] = MSE(target_test, nn.predict(X_test))
            R2_scores[i, j] = R2(target_test, nn.predict(X_test))

    # to account for divergent conditions
    MSE_scores[MSE_scores > 1] = np.nan
    R2_scores[R2_scores > 1] = np.nan
    R2_scores[-1 > R2_scores] = np.nan

    MSE_scores = pd.DataFrame(MSE_scores, columns = etas, index = l2s)
    R2_scores = pd.DataFrame(R2_scores, columns = etas, index = l2s)

    sns.set()
    fig, ax = plt.subplots()

    sns.heatmap(MSE_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'})
    x_ticks = [fr'$10^{{{int(np.log10(eta))}}}$' for eta in etas]
    ax.set_xticklabels(x_ticks)
    y_ticks = [fr'$10^{{{int(np.log10(l2))}}}$' if l2 != 0 else '0' for l2 in l2s]
    ax.set_yticklabels(y_ticks, rotation=0)
    ax.set_xlabel("Initial learning rate $\eta$")
    ax.set_ylabel("$L_2$ regularization parameter")
    ax.add_patch(plt.Rectangle((0, 6), MSE_scores.shape[1], 1, fill=False, edgecolor='black', lw=3))
    plt.tight_layout()
    plt.savefig(f'figures/nn_regression/nn_regression_{learning_method.__name__}_MSE_epochs_{epochs}_batch_size_{batch_size}_act_func_{layer_func.__name__}_Franke_{n}_points.pdf', bbox_inches='tight')

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(R2_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': r'$R^2$ score'})
    x_ticks = [fr'$10^{{{int(np.log10(eta))}}}$' for eta in etas]
    ax.set_xticklabels(x_ticks)
    y_ticks = [fr'$10^{{{int(np.log10(l2))}}}$' if l2 != 0 else '0' for l2 in l2s]
    ax.set_yticklabels(y_ticks, rotation=0)
    ax.set_xlabel("Initial learning rate $\eta$")
    ax.set_ylabel("$L_2$ regularization parameter")
    ax.add_patch(plt.Rectangle((0, 6), R2_scores.shape[1], 1, fill=False, edgecolor='black', lw=3))
    plt.tight_layout()
    plt.savefig(f'figures/nn_regression/nn_regression_{learning_method.__name__}_R2_epochs_{epochs}_batch_size_{batch_size}_act_func_{layer_func.__name__}_Franke_{n}_points.pdf', bbox_inches='tight')
    plt.close(fig='all')

def nn_regression_history():
    np.random.seed(2023)
    rng = np.random.RandomState(2023)
    n = 40
    points = n*n
    x = rng.rand(points)
    y = rng.rand(points)
    z = franke_function(x, y)
    noise = np.random.normal(0, 1, z.shape)*0.1
    z = z + noise
    X = np.array([x, y]).T

    X_train, X_test, target_train, target_test = train_test_split(X, z, test_size=0.2)

    methods = [SGD_const, SGD_AdaGrad, SGD_RMSProp, SGD_ADAM]
    eval_funcs = [MSE, R2]
    epochs = 300
    batch_size = 16
    # l2 = 0
    # l2s = [0]*4
    # l2s = []
    l2s = [1e-2, 1e-4, 1e-3, 1e-5] # best cases
    # eta = 1e-3
    # etas = [1e-2]*4
    # etas = [1e-4]*4
    etas = [1e-3, 1e-1, 1e-2, 1e-2] # best cases
    # etas = [1e-3, 1e-3, 1e-4, 1e-4]
    # histories = np.zeros((len(methods)))
    # constant learning rate
    fig, ax = plt.subplots(nrows=2, sharex=True)
    # for method in
    reg_const = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2s[0],
                                            output_activation_function=identity,
                                            hidden_activation_function=leaky_ReLU,
                                            hidden_activation_derivative=leaky_ReLU_derivative)
    reg_const.train(optimizer=methods[0], init_lr=etas[0], batch_size=batch_size, n_epochs=epochs,
                    evaluation_func=eval_funcs, history=True, t_test=target_test, X_test=X_test)
    ax[0].plot(np.arange(epochs)[:150], reg_const.history[0,:150], label=f'SGD {reg_const.history[0,-1]:12.3f}')
    ax[1].plot(np.arange(epochs)[:150], reg_const.history[1,:150], label='SGD')
    
    reg_ada = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2s[1],
                                            output_activation_function=identity,
                                            hidden_activation_function=leaky_ReLU,
                                            hidden_activation_derivative=leaky_ReLU_derivative)
    reg_ada.train(optimizer=methods[1], init_lr=etas[1], batch_size=batch_size, n_epochs=epochs,
                    evaluation_func=eval_funcs, history=True, t_test=target_test, X_test=X_test)
    ax[0].plot(np.arange(epochs)[:150], reg_ada.history[0, :150], label=f'AdaGrad {reg_ada.history[0,-1]:.3f}')
    ax[1].plot(np.arange(epochs)[:150], reg_ada.history[1, :150], label='AdaGrad')

    reg_rms = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2s[2],
                                            output_activation_function=identity,
                                            hidden_activation_function=leaky_ReLU,
                                            hidden_activation_derivative=leaky_ReLU_derivative)
    reg_rms.train(optimizer=methods[2], init_lr=etas[2], batch_size=batch_size, n_epochs=epochs,
                    evaluation_func=eval_funcs, history=True, t_test=target_test, X_test=X_test)
    ax[0].plot(np.arange(epochs)[:150], reg_rms.history[0, :150], label=f'RMSProp {reg_rms.history[0,-1]:.3f}')
    ax[1].plot(np.arange(epochs)[:150], reg_rms.history[1, :150], label='RMSProp')

    reg_ADAM = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2s[3],
                                            output_activation_function=identity,
                                            hidden_activation_function=leaky_ReLU,
                                            hidden_activation_derivative=leaky_ReLU_derivative)
    reg_ADAM.train(optimizer=methods[3], init_lr=etas[3], batch_size=batch_size, n_epochs=epochs,
                   evaluation_func=eval_funcs, history=True, t_test=target_test, X_test=X_test)
    ax[0].plot(np.arange(epochs)[:150], reg_ADAM.history[0, :150], label=f'ADAM {reg_ADAM.history[0,-1]:10.3f}')
    ax[1].plot(np.arange(epochs)[:150], reg_ADAM.history[1, :150], label='ADAM')

    ax[0].set_ylim(0, 0.2)
    ax[1].set_ylim(0, 1)
    ax[0].set_ylabel('MSE')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('$R^2$')
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(f'figures/nn_regression/behaviour_best_scores_MSE_R2_epochs_{epochs}.pdf', bbox_inches='tight')
    # ax[0].set_title(f'eta {etas[0]}')
    # plt.show()

def nn_classification_network(sklearn=False, layer_func=sigmoid):
    """ Function for executing grid searches for l2 and eta parameter space
        doing logistic regression.
        
        Argument:
            sklearn : (bool) default=False
                Allows to get the accuracy scores using sklearns
                logistic regression method for comparison.
        Returns:
            None
    """
    np.random.seed(2023)
    data = pd.read_csv('data.csv')
    
    diagnosis = data['diagnosis']
    diagnosis_int = (diagnosis == 'M')*1
    predictors = data.drop(['id','diagnosis','Unnamed: 32'], axis='columns')

    X = np.array(predictors)
    target = np.array(diagnosis_int)
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2)
    methods = [SGD_const, SGD_AdaGrad, SGD_ADAM, SGD_RMSProp]
    etas = np.logspace(0, -5, 6)
    n_l2 = 7
    l2s = np.zeros(n_l2)
    l2s[:-1] = np.logspace(0, -5, 6)
    epochs = 300
    batch_size = 16
    layers = 2
    neurons = 30
    accuracies = np.zeros((len(methods), len(l2s), len(etas)))

    for i, method in enumerate(methods):
        print(f'Grid search with {method.__name__} ongoing:')
        for j, l2 in enumerate(l2s):
            for k, eta in enumerate(etas):

                if layer_func == sigmoid:
                    nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=layers, n_hidden_neurons=neurons, L2=l2,
                                                output_activation_function=sigmoid,
                                                hidden_activation_function=sigmoid,
                                                hidden_activation_derivative=sigmoid_derivative)
                elif layer_func == ReLU:
                    nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2,
                                                output_activation_function=sigmoid,
                                                hidden_activation_function=ReLU,
                                                hidden_activation_derivative=ReLU_derivative)
                elif layer_func == leaky_ReLU:
                    nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2,
                                                output_activation_function=sigmoid,
                                                hidden_activation_function=leaky_ReLU,
                                                hidden_activation_derivative=leaky_ReLU_derivative)
            
                # nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=layers, n_hidden_neurons=neurons, L2=l2,
                #                                     output_activation_function=sigmoid, hidden_activation_function=sigmoid,
                #                                     hidden_activation_derivative=sigmoid_derivative)
                nn.train(optimizer=method, init_lr=eta, batch_size=batch_size, n_epochs=epochs)

                accuracies[i, j, k] = accuracy_score(target_test, nn.predict(X_test)) 

    for i, method in enumerate(methods):
        accuracy = pd.DataFrame(accuracies[i], columns = etas, index = l2s)

        sns.set()
        # pylab.rcParams.update(params)
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis",
                    cbar_kws={'label': 'Accuracy'})
        # ax.set_title(f'Logistic regression with:\n{method.__name__} | epochs {epochs} | batch size {batch_size}')
        ax.set_xlabel("Initial learning rate $\eta$")
        ax.set_ylabel("$L_2$ regularization parameter")
        x_ticks = [fr'$10^{{{int(np.log10(eta))}}}$' for eta in etas]
        ax.set_xticklabels(x_ticks)
        y_ticks = [fr'$10^{{{int(np.log10(l2))}}}$' if l2 != 0 else '0' for l2 in l2s]
        ax.set_yticklabels(y_ticks, rotation=0)
        # ax.set_yticklabels(['1e{:.0f}'.format(np.log10(l2)) for l2 in l2s])
        ax.add_patch(plt.Rectangle((0, 6), accuracy.shape[1], 1, fill=False, edgecolor='black', lw=3))
        plt.tight_layout()
        plt.savefig(f'figures/nn_classification/nn_classification_network_{method.__name__}_act_func_{layer_func.__name__}_epochs_{epochs}_batch_size_{batch_size}_layers_{layers}_neurons_{neurons}.pdf', bbox_inches='tight')
        # exit()
        # plt.show()

def nn_classification_network_layers_neurons(learning_method=SGD_const, layer_func=sigmoid):
    data = pd.read_csv('data.csv')
    
    diagnosis = data['diagnosis']
    diagnosis_int = (diagnosis == 'M')*1
    predictors = data.drop(['id','diagnosis','Unnamed: 32'], axis='columns')

    X = np.array(predictors)
    target = np.array(diagnosis_int)
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2)
    methods = [SGD_const, SGD_AdaGrad, SGD_ADAM, SGD_RMSProp]

    # etas = np.logspace(0, -5, 6)
    etas = [1e-4, 1e-3, 1e-3, 1e-3]
    l2s = [1e-3, 1e-5, 1e-3, 1e-4]
    # l2 = np.logspace(0, -5, 6)
    neurons = [5, 10, 30, 50]
    layers = np.linspace(1, 3, 3)
    # batch_sizes = [2**i for i in range(4, 8)]

    accuracies = np.zeros((len(methods), len(layers),len(neurons)))

    for i, method in enumerate(methods):
        for j, l in enumerate(layers):
            for k, n in enumerate(neurons):
                # nn = FeedForwardNeuralNetwork(X, z, n_hidden_layers=int(l), n_hidden_neurons=n, L2=0,
                #                               output_activation_function=identity,
                #                               hidden_activation_function=sigmoid,
                #                               hidden_activation_derivative=sigmoid_derivative)
                
                if layer_func == sigmoid:
                    nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=int(l), n_hidden_neurons=n, L2=l2s[i],
                                                output_activation_function=sigmoid,
                                                hidden_activation_function=sigmoid,
                                                hidden_activation_derivative=sigmoid_derivative)
                elif layer_func == ReLU:
                    nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=int(l), n_hidden_neurons=n, L2=l2s[i],
                                                output_activation_function=sigmoid,
                                                hidden_activation_function=ReLU,
                                                hidden_activation_derivative=ReLU_derivative)
                elif layer_func == leaky_ReLU:
                    nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=int(l), n_hidden_neurons=n, L2=l2s[i],
                                                output_activation_function=sigmoid,
                                                hidden_activation_function=leaky_ReLU,
                                                hidden_activation_derivative=leaky_ReLU_derivative)

                # nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=int(l), n_hidden_neurons=n, L2=l2s[i],
                #                             output_activation_function=sigmoid,
                #                             hidden_activation_function=sigmoid,
                #                             hidden_activation_derivative=sigmoid_derivative)
                nn.train(method, n_epochs=300, init_lr=etas[i], batch_size=16)#len(z))
                accuracies[i, j, k] = accuracy_score(target_test, nn.predict(X_test)) 
        
        accuracy_scores = pd.DataFrame(accuracies[i], columns = neurons, index = layers)

        sns.set()
        fig, ax = plt.subplots()
        sns.heatmap(accuracy_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'})
        # ax.set_title(f'SGD_const MSE')
        ax.set_xlabel("Neurons")
        ax.set_ylabel("Layers")
        plt.tight_layout()
        plt.savefig(f'figures/nn_classification/nn_structure_network_{method.__name__}_l2_{l2s[i]}_eta_{etas[i]}.pdf', bbox_inches='tight')


def logreg_network(sklearn=False):
    """ Function for executing grid searches for l2 and eta parameter space
        doing logistic regression.
        
        Argument:
            sklearn : (bool) default=False
                Allows to get the accuracy scores using sklearns
                logistic regression method for comparison.
                """
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
    np.random.seed(2023)
    diagnosis = data['diagnosis']
    diagnosis_int = (diagnosis == 'M')*1
    predictors = data.drop(['id','diagnosis','Unnamed: 32'], axis='columns')

    X = np.array(predictors)
    target = np.array(diagnosis_int)
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2)
    methods = [SGD_const, SGD_AdaGrad, SGD_ADAM, SGD_RMSProp]
    etas = np.logspace(0, -5, 6)
    n_l2 = 7
    l2s = np.zeros(n_l2)
    l2s[:-1] = np.logspace(0, -5, 6)
    epochs = 300
    batch_size = 16
    accuracies = np.zeros((len(methods), len(l2s), len(etas)))

    if sklearn == True:
        sk_accuracy = np.zeros(len(l2s))
        for i, l2 in enumerate(l2s):
            if l2 == 0:
                logreg = skLogisticRegression(solver='lbfgs', max_iter=epochs, penalty='none')
            else:
                logreg = skLogisticRegression(solver='lbfgs', max_iter=epochs, penalty='l2', C=l2)

            logreg.fit(X_train, target_train)
            sk_accuracy[i] = logreg.score(X_test, target_test)
        print('Accuracies using sklearns Logistic Regression')
        df = pd.DataFrame({'L2 penalty parameter': l2s, 'Accuracy': sk_accuracy})
        table = df.to_markdown(index=False)
        print(table)

    # histories = np.zeros(len)

    for i, method in enumerate(methods):
        print(f'Grid search with {method.__name__} ongoing:')
        for j, l2 in enumerate(l2s):
            for k, eta in enumerate(etas):
                instance = LogisticRegression(X_train, target_train, L2=l2)
                instance.train(optimizer=method, init_lr=eta, batch_size=batch_size, n_epochs=epochs)

                accuracies[i, j, k] = accuracy_score(target_test, instance.predict(X_test)) 

    for i, method in enumerate(methods):
        accuracy = pd.DataFrame(accuracies[i], columns = etas, index = l2s)

        sns.set()
        # pylab.rcParams.update(params)
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis",
                    cbar_kws={'label': 'Accuracy'})
        ax.set_xlabel("Initial learning rate $\eta$")
        ax.set_ylabel("$L_2$ regularization parameter")
        x_ticks = [fr'$10^{{{int(np.log10(eta))}}}$' for eta in etas]
        ax.set_xticklabels(x_ticks)
        y_ticks = [fr'$10^{{{int(np.log10(l2))}}}$' if l2 != 0 else '0' for l2 in l2s]
        ax.set_yticklabels(y_ticks, rotation=0)
        ax.add_patch(plt.Rectangle((0, 6), accuracy.shape[1], 1, fill=False, edgecolor='black', lw=3))
        plt.tight_layout()
        plt.savefig(f'figures/logreg/logreg_network_{method.__name__}_epochs_{epochs}_batch_size_{batch_size}.pdf', bbox_inches='tight')


def logreg_history():
    data = pd.read_csv('data.csv')
    diagnosis = data['diagnosis']
    diagnosis_int = (diagnosis == 'M')*1
    predictors = data.drop(['id','diagnosis','Unnamed: 32'], axis='columns')

    X = np.array(predictors)
    target = np.array(diagnosis_int)
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2)
    methods = [SGD_const, SGD_AdaGrad, SGD_RMSProp, SGD_ADAM]
    epochs = 300
    batch_size = 16
    l2 = 0
    eta = 1e-2
    # histories = np.zeros((len(methods)))
    # constant learning rate
    logreg_const = LogisticRegression(X_train, target_train, L2=l2)
    logreg_const.train(optimizer=methods[0], init_lr=eta, batch_size=batch_size, n_epochs=epochs,
                       history=True, t_test=target_test, X_test=X_test)
    plt.plot(np.arange(epochs), logreg_const.history, label=f'{methods[0].__name__}'.strip('SGD_'))
    # plt.show()
    logreg_AdaGrad = LogisticRegression(X_train, target_train, L2=l2)
    logreg_AdaGrad.train(optimizer=methods[1], init_lr=eta, batch_size=batch_size, n_epochs=epochs,
                       history=True, t_test=target_test, X_test=X_test)
    plt.plot(np.arange(epochs), logreg_AdaGrad.history, label=f'{methods[1].__name__}'.strip('SGD_'))
    # plt.show()
    logreg_RMSProp = LogisticRegression(X_train, target_train, L2=l2)
    logreg_RMSProp.train(optimizer=methods[2], init_lr=eta, batch_size=batch_size, n_epochs=epochs,
                       history=True, t_test=target_test, X_test=X_test)
    plt.plot(np.arange(epochs), logreg_RMSProp.history, label=f'{methods[2].__name__}'.strip('SGD_'))
    # plt.show()
    logreg_ADAM = LogisticRegression(X_train, target_train, L2=l2)
    logreg_ADAM.train(optimizer=methods[3], init_lr=eta, batch_size=batch_size, n_epochs=epochs,
                       history=True, t_test=target_test, X_test=X_test)
    plt.plot(np.arange(epochs), logreg_ADAM.history, label=f'{methods[3].__name__}'.strip('SGD_'))
    plt.legend()
    plt.show()



if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=ConvergenceWarning) # to quell sklearns convergence warnings 
    # nn_regression_network_OLS_batchsize()
    # nn_regression_network_OLS_structure(learning_method=SGD_const)
    # nn_regression_network_eta_l2(learning_method=SGD_const)
    # nn_regression_network_eta_l2(learning_method=SGD_const, layer_func=ReLU)
    # nn_regression_network_eta_l2(learning_method=SGD_const, layer_func=leaky_ReLU)
    # nn_regression_network_eta_l2(learning_method=SGD_AdaGrad)
    # nn_regression_network_eta_l2(learning_method=SGD_AdaGrad, layer_func=ReLU)
    # nn_regression_network_eta_l2(learning_method=SGD_AdaGrad, layer_func=leaky_ReLU)
    # nn_regression_network_eta_l2(learning_method=SGD_RMSProp)
    # nn_regression_network_eta_l2(learning_method=SGD_RMSProp, layer_func=ReLU)
    # nn_regression_network_eta_l2(learning_method=SGD_RMSProp, layer_func=leaky_ReLU)
    # nn_regression_network_eta_l2(learning_method=SGD_ADAM)
    # nn_regression_network_eta_l2(learning_method=SGD_ADAM, layer_func=ReLU)
    # nn_regression_network_eta_l2(learning_method=SGD_ADAM, layer_func=leaky_ReLU)
    nn_regression_history()
    # nn_classification_network()
    # nn_classification_network(layer_func=ReLU)
    # nn_classification_network(layer_func=leaky_ReLU)
    # nn_classification_network_layers_neurons()
    # logreg_network(sklearn=False)
    # logreg_history()
