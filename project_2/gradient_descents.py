import numpy as np

class ProblemConfig:
    """Define optimization problem
    Attributes
    ----------
    features: np.ndarray
        Inputs.
    target: np.ndarray
        Targets.
    cost_grad_func: callable
        Gradient of the cost function w.r.t parameters. Must take arguments
        `features`, `target`, `model_parameters`.
    model_parameters_init: np.ndarray
        Initial model parameters.
    random_seed: int
        seed for random number generator.
    """

    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        cost_grad_func: callable,
        model_parameters_init: np.ndarray,
        random_seed: int,
    ):
        self.features = features
        self.target = target
        self.cost_grad_func = cost_grad_func
        self.model_parameters_init = model_parameters_init
        self.random_seed = random_seed


class ConvergenceConfig:
    """Define convergence for optimization
    Attributes
    ----------
    meta_eval : callable
        meta evaluation func which takes parameters as only arg
    convergence_threshold : float
        Value which `meta_eval` must evaluate to for convergence
    divergence_treshold : float
        Value which `parameters_change` norm may not exceed

    """

    def __init__(
        self,
        meta_eval: callable,
        convergence_threshold: int = 1e-3,
        divergence_treshold: int = 1e3,
    ) -> None:
        self.meta_eval = meta_eval
        self.convergence_threshold = convergence_threshold
        self.divergence_treshold = divergence_treshold


class StochasticGradientDescent:
    """ """

    def __init__(
        self,
        problem_config: ProblemConfig,
        convergence_config: ConvergenceConfig,
        learning_rate: float,
        momentum_parameter: float,
        mini_batch_size: int,
    ):
        self.momentum_parameter = momentum_parameter

        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.n_mini_batches = int(problem_config.target.shape[0] / mini_batch_size)
        self.rng = np.random.default_rng(problem_config.random_seed)
        self.features = problem_config.features
        self.target = problem_config.target
        self.cost_grad_func = problem_config.cost_grad_func
        self.model_parameters_init = problem_config.model_parameters_init
        self.data_indices = np.arange(self.target.shape[0])
        self.convergence = False
        self.meta_eval = convergence_config.meta_eval
        self.convergence_threshold = convergence_config.convergence_threshold
        self.divergence_treshold = convergence_config.divergence_treshold

    def converged(self):
        param_update_norm = np.linalg.norm(self.parameters_update)
        if param_update_norm > self.divergence_treshold:
            raise ValueError(f"Optimization diverged at iteration {self.iteration}")
        return self.meta_eval(self.parameters) < self.convergence_threshold

    def average_gradient(self, model_parameters: np.ndarray):
        """Compute average gradient for `n_minibatches` number of minibatches
        of size `mini_batch_size` without replacement.
        """
        avg_gradient = np.zeros_like(model_parameters, dtype=float)
        for _ in range(self.n_mini_batches):
            mini_batch_indices = self.rng.choice(
                self.data_indices, size=self.mini_batch_size, replace=False
            )
            mini_batch_input = self.features[mini_batch_indices]
            mini_batch_target = self.target[mini_batch_indices]
            avg_gradient += self.cost_grad_func(
                mini_batch_input, mini_batch_target, model_parameters
            )
        return avg_gradient

    def __call__(self, n_iterations_max: int):
        """Advance estiamte for optimal parameters until convergence is reached or
        `n_iterations_max` number of iterations exceeded.
        """
        self.parameters = self.model_parameters_init
        momentum = np.zeros_like(self.parameters)
        for self.iteration in range(1, n_iterations_max + 1):
            momentum = (
                self.momentum_parameter * momentum
                - self.average_gradient(self.parameters) * self.learning_rate
            )
            self.parameters_update = momentum
            self.parameters += self.parameters_update
            # print(np.linalg.norm(self.parameters_update))
            if self.converged():
                self.convergence = True
                break
        return self.parameters

    def __str__(self):
        return f"""{self.__class__.__name__}
Learning rate: {self.learning_rate:.4g}
Momentum parameter: {self.momentum_parameter:.4g}
Mini batch size: {self.mini_batch_size:.4g}
# of Mini batches: {self.n_mini_batches}
# of epochs: {self.iteration}
Converged: {self.convergence}"""


class GradientDescent(StochasticGradientDescent):
    """
    Gradient Descent
    """

    def __init__(
        self,
        problem_config: ProblemConfig,
        convergence_config: ConvergenceConfig,
        learning_rate: float,
        momentum_parameter: float,
    ):
        n_data = problem_config.target.shape[0]
        super().__init__(
            problem_config,
            convergence_config,
            learning_rate,
            momentum_parameter,
            n_data,
        )
        self.n_mini_batches = 1

    def __str__(self):
        return f"""{self.__class__.__name__}
Learning rate: {self.learning_rate:.4g}
Momentum parameter: {self.momentum_parameter:.4g}
# of iterations: {self.iteration}
Converged: {self.convergence}"""


class Adagrad(StochasticGradientDescent):
    """ """

    def __init__(
        self,
        problem_config: ProblemConfig,
        convergence_config: ConvergenceConfig,
        learning_rate: float,
        mini_batch_size: int,
        regularization: float = 1e-8,
    ):
        super().__init__(
            problem_config,
            convergence_config,
            learning_rate,
            None,
            mini_batch_size,
        )
        del self.momentum_parameter
        self.regularization = regularization

    def __call__(self, n_iterations_max: int):
        """Advance estiamte for optimal parameters until convergence is reached or
        `n_iterations_max` number of iterations exceeded.
        """
        self.parameters = self.model_parameters_init
        cumulative_squared_gradient = np.zeros_like(self.model_parameters_init)
        for self.iteration in range(1, n_iterations_max + 1):
            avg_gradient = self.average_gradient(self.parameters)
            cumulative_squared_gradient += avg_gradient * avg_gradient
            self.parameters_update = -(self.learning_rate * avg_gradient) / (
                self.regularization + np.sqrt(cumulative_squared_gradient)
            )
            self.parameters += self.parameters_update
            if self.converged():
                self.convergence = True
                break
        return self.parameters

    def __str__(self):
        return f"""{self.__class__.__name__}
Learning rate: {self.learning_rate:.4g}
Mini batch size: {self.mini_batch_size:.4g}
Regularization constant: {self.regularization:.4g}
# of Mini batches: {self.n_mini_batches}
# of epochs: {self.iteration}
Converged: {self.convergence}"""


class RMSProp(StochasticGradientDescent):
    """
    IDK
    """

    def __init__(
        self,
        problem_config: ProblemConfig,
        convergence_config: ConvergenceConfig,
        learning_rate: float,
        mini_batch_size: int,
        decay_rate: float,
        regularization: float = 1e-8,
    ):
        super().__init__(
            problem_config,
            convergence_config,
            learning_rate,
            None,
            mini_batch_size,
        )
        del self.momentum_parameter
        self.decay_rate = decay_rate
        self.regularization = regularization

    def __call__(self, n_iterations_max: int):
        """Advance estiamte for optimal parameters until convergence is reached or
        `n_iterations_max` number of iterations exceeded.
        """
        self.parameters = self.model_parameters_init
        cumulative_squared_gradient = np.zeros_like(self.model_parameters_init)
        for self.iteration in range(1, n_iterations_max + 1):
            avg_gradient = self.average_gradient(self.parameters)
            cumulative_squared_gradient = (
                self.decay_rate * cumulative_squared_gradient
                + (1 - self.decay_rate) * avg_gradient * avg_gradient
            )
            self.parameters_update = -(self.learning_rate * avg_gradient) / (
                self.regularization + np.sqrt(cumulative_squared_gradient)
            )
            self.parameters += self.parameters_update
            if self.converged():
                self.convergence = True
                break
        return self.parameters

    def __str__(self):
        return f"""{self.__class__.__name__}
Learning rate: {self.learning_rate:.4g}
Regularization constant: {self.regularization:.4g}
Decay rate: {self.decay_rate:.4g}
Mini batch size: {self.mini_batch_size:.4g}
# of mini batches: {self.n_mini_batches}
# of epochs: {self.iteration}
Converged: {self.convergence}"""


class ADAM(StochasticGradientDescent):
    def __init__(
        self,
        problem_config: ProblemConfig,
        convergence_cofig: ConvergenceConfig,
        learning_rate: float,
        mini_batch_size: int,
        momentum_decay_rate: float = 0.9,
        accumulated_decay_rate: float = 0.999,
        regularization: float = 1e-8,
    ):
        super().__init__(
            problem_config,
            convergence_cofig,
            learning_rate,
            None,
            mini_batch_size,
        )
        del self.momentum_parameter
        self.momentum_decay_rate = momentum_decay_rate
        self.accumulated_decay_rate = accumulated_decay_rate
        self.regularization = regularization

    def __call__(self, n_iterations_max: int):
        """Advance estiamte for optimal parameters until convergence is reached or
        `n_iterations_max` number of iterations exceeded.
        """
        self.parameters = self.model_parameters_init
        first_moment = np.zeros_like(self.parameters)
        second_moment = np.zeros_like(self.parameters)
        time_step = 0
        for self.iteration in range(1, n_iterations_max + 1):
            avg_gradient = self.average_gradient(self.parameters)
            time_step += 1
            first_moment = (
                self.momentum_decay_rate * first_moment
                + (1 - self.momentum_decay_rate) * avg_gradient
            )
            second_moment = (
                self.accumulated_decay_rate * second_moment
                + (1 - self.accumulated_decay_rate) * avg_gradient * avg_gradient
            )
            first_moment_corrected = first_moment / (
                1 - self.momentum_decay_rate**time_step
            )
            second_moment_corrected = second_moment / (
                1 - self.accumulated_decay_rate**time_step
            )
            self.parameters_update = (
                -self.learning_rate
                * first_moment_corrected
                / (np.sqrt(second_moment_corrected) + self.regularization)
            )
            self.parameters += self.parameters_update
            if self.converged():
                self.convergence = True
                break
        return self.parameters

    def __str__(self):
        return f"""{self.__class__.__name__}
Learning rate: {self.learning_rate:.4g}
Regularization constant: {self.regularization:.4g}
Momentum decay rate: {self.momentum_decay_rate:.4g}
Accumulated decay rate: {self.accumulated_decay_rate:.4g}
Mini batch size: {self.mini_batch_size:.4g}
# of mini batches: {self.n_mini_batches}
# of epochs: {self.iteration}
Converged: {self.convergence}"""
