
import numpy as np


class SGD_const:
    def __init__(self, X_data, Y_data, 
                gradient_func, init_model_parameters, 
                init_lr, batch_size, momentum = 0.0, random_state=2023):
        self.X_full = X_data
        self.Y_full = Y_data
        self.gradient = gradient_func
        self.n_inputs, self.n_features = np.shape(X_data)
        self.data_indices = np.arange(self.n_inputs)
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.n_iterations = self.n_inputs // batch_size
        self.n_parameters = len(init_model_parameters)
        # Initialize random state
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def advance(self, model_parameters):

        for j in range(self.n_iterations):
            # pick datapoints with replacement
            batch_datapoints = self.rng.choice(self.data_indices, size=self.batch_size, replace=False)
             # set up minibatch with training data
            X = self.X_full[batch_datapoints]
            Y = self.Y_full[batch_datapoints]

            # calculate model parameter gradients in mini batch
            parameter_gradients = self.gradient(X, Y, model_parameters)

            # update model parameters, here using a fixed learning rate
            for i in range(self.n_parameters):
                model_parameters[i] -= self.init_lr * parameter_gradients[i]

        return model_parameters

class SGD_AdaGrad:
    def __init__(self, X_data, Y_data, 
                gradient_func, init_model_parameters, 
                init_lr, batch_size, momentum=0.0,
                random_state=2023):
        self.X_full = X_data
        self.Y_full = Y_data
        self.gradient = gradient_func
        self.n_inputs, self.n_features = np.shape(X_data)
        self.data_indices = np.arange(self.n_inputs)
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.n_iterations = self.n_inputs // batch_size
        self.n_parameters = len(init_model_parameters)

        # Momentum
        self.change = [0.0] * self.n_parameters
        self.momentum = momentum

        # Learning schedule
        self.Giter = [0.0] * self.n_parameters
        self.epsilon = 1e-8

        # Initialize random state
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def advance(self, model_parameters):
        self.Giter = [0.0] * self.n_parameters
        self.change = [0.0] * self.n_parameters

        for j in range(self.n_iterations):
            # pick datapoints with replacement
            batch_datapoints = self.rng.choice(self.data_indices, size=self.batch_size, replace=False)
             # set up minibatch with training data
            X = self.X_full[batch_datapoints]
            Y = self.Y_full[batch_datapoints]

            # calculate model parameter gradients in mini batch
            parameter_gradients = self.gradient(X, Y, model_parameters)

            # update model parameters, here using a fixed learning rate
            for i in range(self.n_parameters):
                self.Giter[i] += parameter_gradients[i] * parameter_gradients[i]
                updated_lr = self.init_lr/(self.epsilon + np.sqrt(self.Giter[i]))
                update = updated_lr * parameter_gradients[i] + self.change[i]*self.momentum
                model_parameters[i] -= update
                self.change[i] = update

        return model_parameters

class SGD_RMSProp:
    def __init__(self, X_data, Y_data, 
                gradient_func, init_model_parameters, 
                init_lr, batch_size,
                momentum=0.0, beta = 0.9, 
                random_state=2023):
        self.X_full = X_data
        self.Y_full = Y_data
        self.gradient = gradient_func
        self.n_inputs, self.n_features = np.shape(X_data)
        self.data_indices = np.arange(self.n_inputs)
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.n_iterations = self.n_inputs // batch_size
        self.n_parameters = len(init_model_parameters)

        # Momentum
        self.change = [0.0] * self.n_parameters
        self.momentum = momentum

        # Learning schedule
        self.Giter = [0.0] * self.n_parameters
        self.beta = beta
        self.epsilon = 1e-8

        # Initialize random state
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def advance(self, model_parameters):
        self.Giter = [0.0] * self.n_parameters
        self.change = [0.0] * self.n_parameters

        for j in range(self.n_iterations):
            # pick datapoints with replacement
            batch_datapoints = self.rng.choice(self.data_indices, size=self.batch_size, replace=False)
             # set up minibatch with training data
            X = self.X_full[batch_datapoints]
            Y = self.Y_full[batch_datapoints]

            # calculate model parameter gradients in mini batch
            parameter_gradients = self.gradient(X, Y, model_parameters)

            # update model parameters, here using a fixed learning rate
            for i in range(self.n_parameters):
                self.Giter[i] = self.beta*self.Giter[i] + (1-self.beta) * parameter_gradients[i] * parameter_gradients[i]
                updated_lr = self.init_lr/(self.epsilon + np.sqrt(self.Giter[i]))
                update = updated_lr * parameter_gradients[i] + self.change[i]*self.momentum
                model_parameters[i] -= update
                self.change[i] = update

        return model_parameters

class SGD_ADAM:
    def __init__(self, X_data, Y_data, 
                gradient_func, init_model_parameters, 
                init_lr, batch_size,
                momentum=0.0, beta = 0.9, rho=0.99,
                random_state=2023):
        self.X_full = X_data
        self.Y_full = Y_data
        self.gradient = gradient_func
        self.n_inputs, self.n_features = np.shape(X_data)
        self.data_indices = np.arange(self.n_inputs)
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.n_iterations = self.n_inputs // batch_size
        self.n_parameters = len(init_model_parameters)

        # Momentum
        self.change = [0.0] * self.n_parameters
        self.momentum = momentum

        # Learning schedule
        self.beta = beta
        self.rho = rho
        self.iter = 0
        self.miter = [0.0] * self.n_parameters
        self.siter = [0.0] * self.n_parameters
        self.epsilon = 1e-8

        # Initialize random state
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def advance(self, model_parameters):
        self.miter = [0.0] * self.n_parameters
        self.siter = [0.0] * self.n_parameters
        self.iter += 1
        for j in range(self.n_iterations):
            # pick datapoints with replacement
            batch_datapoints = self.rng.choice(self.data_indices, size=self.batch_size, replace=False)
             # set up minibatch with training data
            X = self.X_full[batch_datapoints]
            Y = self.Y_full[batch_datapoints]

            # calculate model parameter gradients in mini batch
            parameter_gradients = self.gradient(X, Y, model_parameters)

            # update model parameters, here using a fixed learning rate
            for i in range(self.n_parameters):
                self.miter[i] = (self.beta*self.miter[i] + (1-self.beta)*parameter_gradients[i])/(1-self.beta**self.iter)
                self.siter[i] = (self.rho*self.siter[i] + (1-self.rho)*parameter_gradients[i]*parameter_gradients[i])/(1-self.rho**self.iter)
                updated_lr = self.init_lr/(self.epsilon + np.sqrt(self.siter[i]))
                update = updated_lr * self.miter[i] + self.change[i]*self.momentum
                model_parameters[i] -= update
                self.change[i] = update

        return model_parameters