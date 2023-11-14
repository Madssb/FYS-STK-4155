import numpy as np
from loss_functions import mean_squared_error


class SGDConfig:
    """
    Attributes
    ----------
    learning_rate: float
        (Initial) learning rate, coefficient for grad func w.r.t parameters.
    momentum_parameter: float
        Momentum parameter, coefficient for momentum term.
    n_mini_batches: int
        Number of mini batches.
    mini_batch_size: int
        Number of data per mini batch.
    """
    def __init__(self, learning_rate: float, momentum_parameter: float,
                 n_mini_batches: int, mini_batch_size: int,
                 random_seed: int):
        self.learning_rate = learning_rate
        self.momentum_parameter = momentum_parameter
        self.n_mini_batches = n_mini_batches
        self.mini_batch_size = mini_batch_size
        self.random_seed = random_seed

class StochasticGradientDescent:
    """
    Stochastic Gradient Descent with momentum term.

    Attributes
    ----------
    learning_rate: float
        Learning rate, coefficient for grad func w.r.t parameters.
    momentum_parameter: float
        Momentum parameter, coefficient for momentum term.
    n_mini_batches: int
        Number of mini batches.
    mini_batch_size: int
        Number of data per mini batch.
    rng: Generator
        (random number) Generator object, enforces reproducability.
    features: np.ndarray
        Inputs.
    target: np.ndarray
        Targets.
    cost_grad_func: callable
        gradient of cost function w.r.t parameters. Must take arguments 
        `features`, `target`, `model_parameters`.
    model_parameters_init: np.ndarray
        intial model parameters.
    momentum: np.ndarray
        Momentum of parameters.
    data_indices: np.ndarray
        Data indices.
    
    Methods
    -------

    """

    def __init__(self, config: SGDConfig, features: np.ndarray, 
                 target: np.ndarray, cost_grad_func: callable, 
                 model_parameters_init: np.ndarray):
        self.learning_rate = config.learning_rate
        self.momentum_parameter = config.momentum_parameter
        self.n_mini_batches = config.n_mini_batches
        self.mini_batch_size = config.mini_batch_size
        self.rng = np.random.default_rng(config.random_seed)
        self.features = features
        self.target = target
        self.cost_grad_func = cost_grad_func
        self.model_parameters_init = model_parameters_init
        self.momentum = np.zeros_like(model_parameters_init)
        self.data_indices = np.arange(target.shape[0])

    def average_gradient(self, model_parameters: np.ndarray):
        """
        Compute average gradient for `n_minibatches` number of minibatches
        of size `mini_batch_size` without replacement.
        """
        avg_gradient = np.zeros_like(model_parameters, dtype=float)
        for _ in range(self.n_mini_batches):
            mini_batch_indices = self.rng.choice(self.data_indices, 
                                                size=self.mini_batch_size,
                                                replace=False)
            mini_batch_input = self.features[mini_batch_indices]
            mini_batch_target = self.target[mini_batch_indices]
            avg_gradient += self.cost_grad_func(mini_batch_input, 
                                                mini_batch_target,
                                                model_parameters)
        return avg_gradient

    def advance(self, model_parameters: np.ndarray):
        """
        Advance estimate for optimal parameters with a single step with fixed
        learning rate, and momentum.

        Returns
        -------
        new_model_parameters: np.ndarray
            Updated model parameters.
        """
        self.momentum = self.momentum_parameter*self.momentum \
            + self.average_gradient(model_parameters) * self.learning_rate
        new_model_parameters = model_parameters - self.momentum
        return new_model_parameters
    
    def __call__(self, n_iterations_max: int, tolerance: float):
        """
        Advance estiamte for optimal parameters until convergence is reached or
        `n_iterations_max` number of iterations exceeded.
        """
        parameters = self.model_parameters_init
        for iteration in range(n_iterations_max):
            parameters_new = self.advance(parameters)
            if np.linalg.norm(parameters_new - parameters) < tolerance:
                parameters = parameters_new
                break
            parameters = parameters_new
        print(f"# of iterations: {iteration}")
        return parameters#, iteration
    
    def __str__(self):
        print(f"""
learning rate: {self.learning_rate:.4g}
momentum parameter: {self.momentum_parameter:.4g}
              """)


class Adagrad:
    """
    x
    """
    def __init__(self, config: SGDConfig, features: np.ndarray, 
                 target: np.ndarray, cost_grad_func: callable, 
                 model_parameters_init: np.ndarray):
        self.learning_rate = config.learning_rate
        self.n_mini_batches = config.n_mini_batches
        self.mini_batch_size = config.mini_batch_size
        self.rng = np.random.default_rng(config.random_seed)
        self.features = features
        self.target = target
        self.cost_grad_func = cost_grad_func  
        self.model_parameters_init = model_parameters_init 
        self.data_indices = np.arange(target.shape[0])  
        self.small_const = 1e-12

    def average_gradient(self, model_parameters: np.ndarray):
        """
        Compute average gradient for `n_minibatches` number of minibatches
        of size `mini_batch_size` without replacement.
        """
        avg_gradient = np.zeros_like(model_parameters, dtype=float)
        for _ in range(self.n_mini_batches):
            mini_batch_indices = self.rng.choice(self.data_indices, 
                                                size=self.mini_batch_size,
                                                replace=False)
            mini_batch_input = self.features[mini_batch_indices]
            mini_batch_target = self.target[mini_batch_indices]
            avg_gradient += self.cost_grad_func(mini_batch_input, 
                                                mini_batch_target,
                                                model_parameters)
        return avg_gradient

    def __call__(self, n_iterations_max: int, tolerance: float):
        last_20_param_updates = np.ones(20)
        parameters = self.model_parameters_init
        cumulative_squared_gradient = np.zeros_like(self.model_parameters_init)
        for iteration in range(n_iterations_max):
            avg_gradient = self.average_gradient(parameters)
            cumulative_squared_gradient += avg_gradient*avg_gradient
            parameters_update = (self.small_const * cumulative_squared_gradient)/(self.small_const + np.sqrt(cumulative_squared_gradient))
            parameters += parameters_update
            last_20_param_updates[iteration % 20] = np.linalg.norm(parameters_update)
            print(f"iteration {iteration}, grad norm: {np.linalg.norm(parameters_update):.4g}")
            if np.linalg.norm(parameters_update)< tolerance:
                break
            if np.mean(last_20_param_updates) < tolerance:
                break
        return parameters

class RMSProp:
    def __init__(self, input, target, 
                gradient_func, model_parameters_init, 
                init_learning_rate, mini_batch_size,
                momentum=0.0, beta = 0.9, 
                random_seed=2023):
        self.input = input
        self.target = target
        self.gradient = gradient_func
        self.n_inputs, self.n_features = np.shape(input)
        self.data_indices = np.arange(self.n_inputs)
        self.init_learning_rate = init_learning_rate
        self.mini_batch_size = mini_batch_size
        self.n_iterations = self.n_inputs // mini_batch_size
        self.n_parameters = len(model_parameters_init)

        # Momentum
        self.change = [0.0] * self.n_parameters
        self.momentum = momentum

        # Learning schedule
        self.Giter = [0.0] * self.n_parameters
        self.beta = beta
        self.epsilon = 1e-8

        # Initialize random state
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def advance(self, model_parameters):
        self.Giter = [0.0] * self.n_parameters
        self.change = [0.0] * self.n_parameters

        for j in range(self.n_iterations):
            # pick datapoints with replacement
            batch_datapoints = self.rng.choice(self.data_indices, size=self.mini_batch_size, replace=False)
             # set up minibatch with training data
            X = self.input[batch_datapoints]
            Y = self.target[batch_datapoints]

            # calculate model parameter gradients in mini batch
            parameter_gradients = self.gradient(X, Y, model_parameters)

            # update model parameters, here using a fixed learning rate
            for i in range(self.n_parameters):
                self.Giter[i] = self.beta*self.Giter[i] + (1-self.beta) * parameter_gradients[i] * parameter_gradients[i]
                updated_lr = self.init_learning_rate/(self.epsilon + np.sqrt(self.Giter[i]))
                update = updated_lr * parameter_gradients[i] + self.change[i]*self.momentum
                model_parameters[i] -= update
                self.change[i] = update

        return model_parameters

class ADAM:
    def __init__(self, input, target, 
                gradient_func, model_parameters_init, 
                init_learning_rate, mini_batch_size,
                momentum=0.0, beta = 0.9, rho=0.99,
                random_seed=2023):
        self.input = input
        self.target = target
        self.gradient = gradient_func
        self.n_inputs, self.n_features = np.shape(input)
        self.data_indices = np.arange(self.n_inputs)
        self.init_learning_rate = init_learning_rate
        self.mini_batch_size = mini_batch_size
        self.n_iterations = self.n_inputs // mini_batch_size
        self.n_parameters = len(model_parameters_init)

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
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def advance(self, model_parameters):
        self.miter = [0.0] * self.n_parameters
        self.siter = [0.0] * self.n_parameters
        self.iter += 1
        for j in range(self.n_iterations):
            # pick datapoints with replacement
            batch_datapoints = self.rng.choice(self.data_indices, size=self.mini_batch_size, replace=False)
             # set up minibatch with training data
            X = self.input[batch_datapoints]
            Y = self.target[batch_datapoints]

            # calculate model parameter gradients in mini batch
            parameter_gradients = self.gradient(X, Y, model_parameters)

            # update model parameters, here using a fixed learning rate
            for i in range(self.n_parameters):
                self.miter[i] = (self.beta*self.miter[i] + (1-self.beta)*parameter_gradients[i])/(1-self.beta**self.iter)
                self.siter[i] = (self.rho*self.siter[i] + (1-self.rho)*parameter_gradients[i]*parameter_gradients[i])/(1-self.rho**self.iter)
                updated_lr = self.init_learning_rate/(self.epsilon + np.sqrt(self.siter[i]))
                update = updated_lr * self.miter[i] + self.change[i]*self.momentum
                model_parameters[i] -= update
                self.change[i] = update

        return model_parameters