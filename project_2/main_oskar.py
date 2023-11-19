# Import external packages
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import warnings
import seaborn as sns

# Import self-made packages
from nn_class import FeedForwardNeuralNetwork
from nn_class import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative, identity
from nn_class import hard_classifier, indicator, accuracy_score, MSE, R2
from SGD import SGD_const, SGD_AdaGrad, SGD_RMSProp, SGD_ADAM
from logreg_gradient_class import LogisticRegression
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from FrankeFunction import franke_function, features_polynomial_2d
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


# x = np.linspace(-10, 10, 100)
x = np.expand_dims(np.linspace(-10, 10, 100), 1)
y = 2 + 3 * x + 4 * x**2
X = np.array([x, x**2]).T
# print(np.shape(x),np.shape(y), np.shape(X))
# exit()

def gradient_decent_plots(epochs=100):
    # franke function
    points = 100
    x = np.arange(0, 1, 1/points)
    y = np.arange(0, 1, 1/points) 
    x_mesh, y_mesh = np.meshgrid(x, y)
    datagrid = [x_mesh, y_mesh]
    analytic = franke_function(x_mesh, y_mesh)
    noise = np.random.normal(0, 1, x_mesh.shape)*0.1 # dampened noise
    target = (analytic + noise).ravel()
    features = features_polynomial_2d(x, y, degree=10)
    # degree = 10
    # poly = PolynomialFeatures(degree=degree)
    # features = poly.fit_transform(np.concatenate(np.stack([datagrid[i] for i in range(0, len(datagrid))], axis=-1), axis=0))
    # print(np.shape(features))
    # exit()

    X_train, X_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

    # for epoch in range(epochs):
    #     optimizer = SGD_const(X_train, f_train, gradient_func=)


    nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=20, L2=0.001,
                                output_activation_function=identity, hidden_activation_function=sigmoid, hidden_activation_derivative=sigmoid_derivative)
    nn.train(SGD_const, evaluation_func=MSE, n_epochs=50, batch_size=len(X_train), init_lr=0.1)

    plt.plot(x, y, label='real')
    plt.plot(x, nn.predict(x), label='model')
    plt.legend()
    plt.show()

def nn_regression_network_OLS(learning_method=SGD_const):
    rng = np.random.RandomState(2023)
    n = 40
    points = n*n
    x = rng.rand(points)
    y = rng.rand(points)
    # x = np.arange(0, 1, 1/points)
    # y = np.arange(0, 1, 1/points)
    z = franke_function(x, y)
    X = np.array([x, y]).T

    X_train, X_test, target_train, target_test = train_test_split(X, z, test_size=0.2)

    etas = np.logspace(0, -5, 6)
    # l2 = np.logspace(0, -5, 6)
    neurons = [5, 10, 30, 50]
    layers = np.linspace(1, 3, 3)
    batch_sizes = [2**i for i in range(4, 8)]

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
            nn.train(learning_method, n_epochs=300, init_lr=0.01, batch_size=16)#len(z))
            MSE_lay_neur[i, j] = MSE(target_train, nn.predict(X_train))
            R2_lay_neur[i, j] = R2(target_train, nn.predict(X_train))
    
    MSE_scores = pd.DataFrame(MSE_lay_neur, columns = neurons, index = layers)
    R2_scores = pd.DataFrame(R2_lay_neur, columns = neurons, index = layers)

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(MSE_scores, annot=True, ax=ax, cmap="viridis")
    ax.set_title(f'SGD_const MSE')
    ax.set_xlabel("$neurons$")
    ax.set_ylabel("$layers$")

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(R2_scores, annot=True, ax=ax, cmap="viridis")
    ax.set_title(f'SGD_const R2')
    ax.set_xlabel("$neurons$")
    ax.set_ylabel("$layers$")

    plt.show()

def nn_regression_network_eta_l2(learning_method=SGD_const, layer_func=sigmoid):
    rng = np.random.RandomState(2023)
    n = 40
    points = n*n
    x = rng.rand(points)
    y = rng.rand(points)
    # x = np.arange(0, 1, 1/points)
    # y = np.arange(0, 1, 1/points)
    z = franke_function(x, y)
    X = np.array([x, y]).T

    X_train, X_test, target_train, target_test = train_test_split(X, z, test_size=0.2)

    etas = np.logspace(0, -5, 6)
    n_l2 = 7
    l2s = np.zeros(n_l2)
    l2s[:-1] = np.logspace(0, -5, 6)
    # print(l2s)
    # exit()
    # neurons = [5, 10, 30, 50]
    # layers = np.linspace(1, 3, 3)
    # batch_sizes = [2**i for i in range(4, 8)]

    MSE_scores = np.zeros((len(l2s),len(etas)))
    R2_scores = np.zeros((len(l2s),len(etas)))

    # constant learning rate

    for i, l2 in enumerate(l2s):
        for j, eta in enumerate(etas):
            # nn = FeedForwardNeuralNetwork(X, z, n_hidden_layers=int(l), n_hidden_neurons=n, L2=0,
            #                               output_activation_function=identity,
            #                               hidden_activation_function=sigmoid,
            #                               hidden_activation_derivative=sigmoid_derivative)
            nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=2, n_hidden_neurons=30, L2=l2,
                                          output_activation_function=identity,
                                          hidden_activation_function=sigmoid,
                                          hidden_activation_derivative=sigmoid_derivative)
            nn.train(optimizer=learning_method, n_epochs=300, init_lr=eta, batch_size=16)#len(z))
            MSE_scores[i, j] = MSE(target_test, nn.predict(X_test))
            R2_scores[i, j] = R2(target_test, nn.predict(X_test))

    # to account for divergent conditions
    MSE_scores[MSE_scores > 1] = np.nan
    R2_scores[R2_scores > 1] = np.nan
    R2_scores[1e-2 > R2_scores] = np.nan

    MSE_scores = pd.DataFrame(MSE_scores, columns = etas, index = l2s)
    R2_scores = pd.DataFrame(R2_scores, columns = etas, index = l2s)

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(MSE_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'})
    ax.set_title(f'SGD_const MSE')
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("$\lambda$")

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(R2_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': r'$R^2$ score'})
    ax.set_title(f'SGD_const R2')
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("$\lambda$")

    plt.show()
    plt.close(fig='all')



    # plt.savefig('figures/nn_classification/train_accuracy_ReLU.pdf')

def nn_classification_network(sklearn=False):
    """ Function for executing grid searches for l2 and eta parameter space
        doing logistic regression.
        
        Argument:
            sklearn : (bool) default=False
                Allows to get the accuracy scores using sklearns
                logistic regression method for comparison.
                """
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
    batch_size = 32
    layers = 2
    neurons = 30
    accuracies = np.zeros((len(methods), len(l2s), len(etas)))

    # if sklearn == True:
    #     sk_accuracy = np.zeros(len(l2s))
    #     for i, l2 in enumerate(l2s):
    #         if l2 == 0:
    #             logreg = skLogisticRegression(solver='lbfgs', max_iter=1000)
    #         else:
    #             logreg = skLogisticRegression(solver='lbfgs', max_iter=1000, penalty='l2', C=l2)

    #         logreg.fit(X_train, target_train)
    #         sk_accuracy[i] = logreg.score(X_test, target_test)
    #     print('Accuracies using sklearns Logistic Regression')
    #     df = pd.DataFrame({'L2 penalty parameter': l2s, 'Accuracy': sk_accuracy})
    #     table = df.to_markdown(index=False)
    #     print(table)

    # histories = np.zeros(len)

    for i, method in enumerate(methods):
        print(f'Grid search with {method.__name__} ongoing:')
        for j, l2 in enumerate(l2s):
            for k, eta in enumerate(etas):
                instance = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=layers, n_hidden_neurons=neurons, L2=l2,
                                                    output_activation_function=sigmoid, hidden_activation_function=sigmoid,
                                                    hidden_activation_derivative=sigmoid_derivative)
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
        # ax.set_title(f'Logistic regression with:\n{method.__name__} | epochs {epochs} | batch size {batch_size}')
        ax.set_xlabel("Initial learning rate $\eta$")
        ax.set_ylabel("$L_2$ regularization parameter")
        # ax.set_yscale("symlog")
        # ax.set_xscale("symlog")
        # ax.ticklabel_format(style='scientific')
        # ax.set_yticks(f'{l2s:e}')
        x_ticks = [fr'$10^{{{int(np.log10(eta))}}}$' for eta in etas]
        ax.set_xticklabels(x_ticks)
        y_ticks = [fr'$10^{{{int(np.log10(l2))}}}$' if l2 != 0 else '0' for l2 in l2s]
        ax.set_yticklabels(y_ticks, rotation=0)
        # ax.set_yticklabels(['1e{:.0f}'.format(np.log10(l2)) for l2 in l2s])
        ax.add_patch(plt.Rectangle((0, 6), accuracy.shape[1], 1, fill=False, edgecolor='black', lw=3))
        plt.tight_layout()
        plt.savefig(f'figures/nn_classification/{method.__name__}_nn_classification_network_epochs_{epochs}_batch_size_{batch_size}_layers_{layers}_neurons_{neurons}.pdf', bbox_inches='tight')
        # exit()
        # plt.show()

def nn_classification_network_layers_neurons(learning_method=SGD_const):
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
    # constant learning rate

    for i, method in enumerate(methods):
        for j, l in enumerate(layers):
            for k, n in enumerate(neurons):
                # nn = FeedForwardNeuralNetwork(X, z, n_hidden_layers=int(l), n_hidden_neurons=n, L2=0,
                #                               output_activation_function=identity,
                #                               hidden_activation_function=sigmoid,
                #                               hidden_activation_derivative=sigmoid_derivative)
                nn = FeedForwardNeuralNetwork(X_train, target_train, n_hidden_layers=int(l), n_hidden_neurons=n, L2=l2s[i],
                                            output_activation_function=sigmoid,
                                            hidden_activation_function=sigmoid,
                                            hidden_activation_derivative=sigmoid_derivative)
                nn.train(method, n_epochs=300, init_lr=etas[i], batch_size=16)#len(z))
                accuracies[i, j, k] = accuracy_score(target_test, nn.predict(X_test)) 
        
        accuracy_scores = pd.DataFrame(accuracies[i], columns = neurons, index = layers)

        sns.set()
        fig, ax = plt.subplots()
        sns.heatmap(accuracy_scores, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'})
        # ax.set_title(f'SGD_const MSE')
        ax.set_xlabel("neurons")
        ax.set_ylabel("layers")
        plt.tight_layout()
        plt.savefig(f'figures/nn_classification/{method.__name__}_nn_structure_network_l2_{l2s[i]}_eta_{etas[i]}.pdf', bbox_inches='tight')


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
    epochs = 500
    batch_size = 16
    accuracies = np.zeros((len(methods), len(l2s), len(etas)))

    if sklearn == True:
        sk_accuracy = np.zeros(len(l2s))
        for i, l2 in enumerate(l2s):
            if l2 == 0:
                logreg = skLogisticRegression(solver='lbfgs', max_iter=1000)
            else:
                logreg = skLogisticRegression(solver='lbfgs', max_iter=1000, penalty='l2', C=l2)

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
        # ax.set_title(f'Logistic regression with:\n{method.__name__} | epochs {epochs} | batch size {batch_size}')
        ax.set_xlabel("Initial learning rate $\eta$")
        ax.set_ylabel("Regularization parameter $\lambda$")
        # ax.set_yscale("symlog")
        # ax.set_xscale("symlog")
        # ax.ticklabel_format(style='scientific')
        # ax.set_yticks(f'{l2s:e}')
        x_ticks = [fr'$10^{{{int(np.log10(eta))}}}$' for eta in etas]
        ax.set_xticklabels(x_ticks)
        y_ticks = [fr'$10^{{{int(np.log10(l2))}}}$' if l2 != 0 else '0' for l2 in l2s]
        ax.set_yticklabels(y_ticks, rotation=0)
        # ax.set_yticklabels(['1e{:.0f}'.format(np.log10(l2)) for l2 in l2s])
        ax.add_patch(plt.Rectangle((0, 6), accuracy.shape[1], 1, fill=False, edgecolor='black', lw=3))
        plt.tight_layout()
        plt.savefig(f'figures/logreg/corr_{method.__name__}_logreg_network_epochs_{epochs}_batch_size_{batch_size}.pdf', bbox_inches='tight')
        # exit()
        # plt.show()

def logreg_history():
    data = pd.read_csv('data.csv')
    diagnosis = data['diagnosis']
    diagnosis_int = (diagnosis == 'M')*1
    predictors = data.drop(['id','diagnosis','Unnamed: 32'], axis='columns')

    X = np.array(predictors)
    target = np.array(diagnosis_int)
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2)
    methods = [SGD_const, SGD_AdaGrad, SGD_RMSProp, SGD_ADAM]
    epochs = 100
    batch_size = 128
    l2 = 1e-5
    eta = 1e-2
    histories = np.zeros((len(methods)))
    # constant learning rate
    logreg_const = LogisticRegression(X_train, target_train, L2=l2)
    logreg_const.train(optimizer=methods[0], init_lr=eta, batch_size=batch_size, n_epochs=epochs,
                       history=True, t_test=target_test, X_test=X_test)
    plt.plot(np.arange(epochs), logreg_const.history)#, label=f'{}')
    # plt.show()
    logreg_AdaGrad = LogisticRegression(X_train, target_train, L2=l2)
    logreg_AdaGrad.train(optimizer=methods[1], init_lr=eta, batch_size=batch_size, n_epochs=epochs,
                       history=True, t_test=target_test, X_test=X_test)
    plt.plot(np.arange(epochs), logreg_AdaGrad.history)
    # plt.show()
    logreg_RMSProp = LogisticRegression(X_train, target_train, L2=l2)
    logreg_RMSProp.train(optimizer=methods[2], init_lr=eta, batch_size=batch_size, n_epochs=epochs,
                       history=True, t_test=target_test, X_test=X_test)
    plt.plot(np.arange(epochs), logreg_RMSProp.history)
    # plt.show()
    logreg_ADAM = LogisticRegression(X_train, target_train, L2=l2)
    logreg_ADAM.train(optimizer=methods[3], init_lr=eta, batch_size=batch_size, n_epochs=epochs,
                       history=True, t_test=target_test, X_test=X_test)
    plt.plot(np.arange(epochs), logreg_ADAM.history)
    plt.show()



if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=ConvergenceWarning) # to quell sklearn lasso's convergence warnings 

    # gradient_decent_plots()
    # nn_regression_network_OLS(learning_method=SGD_const)
    nn_regression_network_eta_l2(learning_method=SGD_const)
    # nn_classification_network()
    # nn_classification_network_layers_neurons()
    # logreg_network(sklearn=False)
    # logreg_history()
