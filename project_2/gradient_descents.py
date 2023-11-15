import numpy as np

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
    Stochastic Gradient Descent.

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
        self.data_indices = np.arange(target.shape[0])
        self.convergence_status = "optimizer not yet called"

    def average_gradient(self, model_parameters: np.ndarray):
        """Compute average gradient for `n_minibatches` number of minibatches
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
        """Advance estiamte for optimal parameters until convergence is reached or
        `n_iterations_max` number of iterations exceeded.
        """
        parameters = self.model_parameters_init
        momentum = np.zeros_like(parameters)
        converged = False
        for iteration in range(1,n_iterations_max+1):
            momentum = self.momentum_parameter*momentum + self.average_gradient(parameters) * self.learning_rate
            parameters = parameters - momentum
            if np.linalg.norm(momentum) < tolerance:
                self.convergence_status = f"{self.__class__.__name__} converged at {iteration} iterations."
                converged = True
                break
        if not converged:
            self.convergence_status = f"{self.__class__.__name__} failed convergence in {n_iterations_max} iterations."
        return parameters
    
    def __str__(self):
        return f"""{self.__class__.__name__}
Learning rate: {self.learning_rate:.4g}
Momentum parameter: {self.momentum_parameter:.4g}
Mini batch size: {self.mini_batch_size:.4g}
# of Mini batches: {self.n_mini_batches}
{self.convergence_status}"""


class Adagrad(StochasticGradientDescent):
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
        self.small_const = 1e-8

    def __call__(self, n_iterations_max: int, tolerance: float):
        """Advance estiamte for optimal parameters until convergence is reached or
        `n_iterations_max` number of iterations exceeded.
        """
        parameters = self.model_parameters_init
        cumulative_squared_gradient = np.zeros_like(self.model_parameters_init)
        converged = False
        for iteration in range(1,n_iterations_max+1):
            avg_gradient = self.average_gradient(parameters)
            cumulative_squared_gradient += avg_gradient * avg_gradient
            parameters_update = -(self.learning_rate * avg_gradient)/(self.small_const + np.sqrt(cumulative_squared_gradient))
            parameters += parameters_update
            if np.linalg.norm(parameters_update) < tolerance:
                self.convergence_status = f"{self.__class__.__name__} converged at {iteration} iterations."
                converged = True
                break
        if not converged:
            self.convergence_status = f"{self.__class__.__name__} failed convergence in {n_iterations_max} iterations."
        return parameters

    def __str__(self):
        return f"""{self.__class__.__name__}
Learning rate: {self.learning_rate:.4g}
Mini batch size: {self.mini_batch_size:.4g}
Small constant: {self.small_const:.4g}
# of Mini batches: {self.n_mini_batches}
{self.convergence_status}"""

class RMSProp(StochasticGradientDescent):   
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
        self.decay_rate = 0.5
        self.small_const = 1e-6

    def __call__(self, n_iterations_max: int, tolerance: float):
        """Advance estiamte for optimal parameters until convergence is reached or
        `n_iterations_max` number of iterations exceeded.
        """
        parameters = self.model_parameters_init
        cumulative_squared_gradient = np.zeros_like(self.model_parameters_init)
        converged = False
        for iteration in range(1,n_iterations_max+1):
            
            avg_gradient = self.average_gradient(parameters)
            cumulative_squared_gradient = self.decay_rate * cumulative_squared_gradient + (1 - self.decay_rate) * avg_gradient * avg_gradient
            parameters_update = -(self.learning_rate * avg_gradient)/(self.small_const + np.sqrt(cumulative_squared_gradient))
            parameters += parameters_update
            if np.linalg.norm(parameters_update) < tolerance:
                self.convergence_status = f"{self.__class__.__name__} converged at {iteration} iterations."
                converged = True
                break
        if not converged:
            self.convergence_status = f"{self.__class__.__name__} failed convergence in {n_iterations_max} iterations."
        return parameters

    def __str__(self):
        return f"""{self.__class__.__name__}
Learning rate: {self.learning_rate:.4g}
Small constant: {self.small_const:.4g}
Decay rate: {self.decay_rate:.4g}
Mini batch size: {self.mini_batch_size:.4g}
# of mini batches: {self.n_mini_batches}
{self.convergence_status}"""

class ADAM(StochasticGradientDescent):
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
        self.small_const = 1e-6
        # rho_1
        self.momentum_decay_rate = 0.9
        # rho_2
        self.accumulated_decay_rate = 0.999

    def __call__(self, n_iterations_max: int, tolerance: float):
        """Advance estiamte for optimal parameters until convergence is reached or
        `n_iterations_max` number of iterations exceeded.
        """
        parameters = self.model_parameters_init
        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        time_step = 0
        converged = False
        for iteration in range(1,n_iterations_max+1):
            avg_gradient = self.average_gradient(parameters)
            time_step += 1
            first_moment = self.momentum_decay_rate * first_moment + (1 - self.momentum_decay_rate)*avg_gradient
            second_moment = self.accumulated_decay_rate * second_moment + (1 - self.accumulated_decay_rate)*avg_gradient*avg_gradient
            first_moment_corrected = first_moment / (1 - self.momentum_decay_rate**time_step)
            second_moment_corrected = second_moment / (1 - self.accumulated_decay_rate**time_step)
            parameters_update = -self.learning_rate*first_moment_corrected/(np.sqrt(second_moment_corrected) + self.small_const)
            parameters += parameters_update
            if np.linalg.norm(parameters_update) < tolerance:
                self.convergence_status = f"{self.__class__.__name__} converged at {iteration} iterations."
                converged = True
                break
        if not converged:
            self.convergence_status = f"{self.__class__.__name__} failed convergence in {n_iterations_max} iterations."
        return parameters

    def __str__(self):
        return f"""{self.__class__.__name__}
Learning rate: {self.learning_rate:.4g}
Small constant: {self.small_const:.4g}
Momentum decay rate: {self.momentum_decay_rate:.4g}
Accumulated decay rate: {self.accumulated_decay_rate:.4g}
Mini batch size: {self.mini_batch_size:.4g}
# of mini batches: {self.n_mini_batches}
{self.convergence_status}"""
