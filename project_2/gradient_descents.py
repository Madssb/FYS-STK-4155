
import numpy as np
from loss_functions import mean_squared_error

class GradientDescent:
    """
    Find local minima with Gradient Descent.

    Parameters
    ----------
    learning_rate : float
        Learning rate.
    grad_func : callable
        Gradient of the function to be optimized.
    init_guess : float, tuple, or numpy.ndarray
        Initial guess for the minima input.
    """

    def __init__(self, learning_rate: float, grad_func: callable,
                 init_guess: float | np.ndarray | tuple):
        if not isinstance(learning_rate, float):
            raise TypeError(
                f"learning_rate: expected float, got {type(learning_rate)}.")
        if not callable(grad_func):
            raise TypeError(
                f"grad_func: expected callable, got {type(grad_func)}.")
        if not isinstance(init_guess, (tuple, float, np.ndarray)):
            raise TypeError(
                f"init_guess: expected tuple, got {type(init_guess)}.")
        self.learning_rate = learning_rate
        self.grad_func = grad_func
        self.init_guess = init_guess

    def advance(self, guess):
        """
        Advance minima value guess with momentum.

        Parameters
        ----------
        guess : float, tuple, or numpy.ndarray
            Current guess for the minima.

        Returns
        -------
        float, tuple, or numpy.ndarray
            Updated minima value guess.
        """
        return guess - self.learning_rate*self.grad_func(guess)

    def __call__(self, iterations_max: int, tolerance: float):
        """
        Compute minima estimate.

        Parameters
        ----------
        iterations_max : int
            Maximum number of iterations.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        float, tuple, or numpy.ndarray
            Estimated minima.
        int
            Number of iterations performed.
        """
        if not isinstance(iterations_max, int):
            raise TypeError(
                f"iterations: expected int, got {type(iterations_max)}")
        if not isinstance(tolerance, float):
            raise TypeError(
                f"tolerance: expected float, got {type(tolerance)}")
        iteration = 0
        guess = self.init_guess
        while True:
            new_guess = self.advance(guess)
            change = new_guess - guess
            iteration += 1
            guess = new_guess
            if iteration >= iterations_max or np.abs(change).all() < tolerance:
                break
        n_iterations = iteration
        return guess, n_iterations


def test_gradient_descent():
    """
    Validate GradientDescent with simple polynomial example.
    """
    def simple_polynomial(x: np.ndarray | float):
        """
        Compute a simple polynomial function.

        Parameters
        ----------
        x : float or numpy.ndarray
            Input values.

        Returns
        -------
        float or numpy.ndarray
            Function values.
        """
        return 5 + 2*x + 7*x**2

    def simple_polynomial_grad(x: np.ndarray | float):
        """
        Compute the gradient of a simple polynomial function.

        Parameters
        ----------
        x : float or numpy.ndarray
            Input values.

        Returns
        -------
        float or numpy.ndarray
            Gradient values.
        """
        return 2 + 14*x

    optimizer = GradientDescent(0.1, simple_polynomial_grad, 2.)
    guess, n_iterations = optimizer(10000, 1e-5)
    domain = np.linspace(-5, 5, 100)
    eval = simple_polynomial(domain)
    minima_point_analytical = -0.1429
    tolerance = 1e-3
    assert np.abs(guess-minima_point_analytical) < tolerance


class GradientDescentWithMomentum(GradientDescent):
    """
    Find local minima with Gradient Descent with Momentum.

    Parameters
    ----------
    learning_rate : float
        Learning rate.
    grad_func : callable
        Gradient of the function to be optimized.
    init_guess : float, tuple, or numpy.ndarray
        Initial guess for the minima input.
    momentum_parameter : float
        Momentum parameter.
    """

    def __init__(self, learning_rate: float, grad_func: callable,
                 init_guess: float | np.ndarray | tuple, momentum_parameter: float):
        # Call the constructor of the parent class (GradientDescent)
        super().__init__(learning_rate, grad_func, init_guess)

        if not isinstance(momentum_parameter, float):
            raise TypeError(
                f"momentum_parameter: expected float, got {type(momentum_parameter)}.")

        self.momentum_parameter = momentum_parameter
        self.momentum = 0  # Initialize momentum to zero

    def advance(self, guess):
        """
        Advance minima value guess with momentum.

        Parameters
        ----------
        guess : float, tuple, or numpy.ndarray
            Current guess for the minima.

        Returns
        -------
        float, tuple, or numpy.ndarray
            Updated minima value guess.
        """
        gradient = self.grad_func(guess)
        self.momentum = self.momentum_parameter * \
            self.momentum + self.learning_rate * gradient
        return guess - self.momentum


def test_gradient_descent_with_momentum():
    """
    Validate GradientDescentWithMomentum with simple polynomial example.
    """
    def simple_polynomial(x: np.ndarray | float) -> np.ndarray | float:
        return 5 + 2 * x + 7 * x ** 2

    def simple_polynomial_grad(x: np.ndarray | float) -> np.ndarray | float:
        return 2 + 14 * x

    optimizer = GradientDescentWithMomentum(
        0.1, simple_polynomial_grad, 2., 0.9)  # Add momentum parameter
    guess, n_iterations = optimizer(10000, 1e-5)
    domain = np.linspace(-5, 5, 100)
    eval = simple_polynomial(domain)
    minima_point_analytical = -0.1429
    tolerance = 1e-3
    assert np.abs(guess - minima_point_analytical) < tolerance


class StochasticGradientDescent(GradientDescent):
    """
    Stochastic Gradient Descent (SGD) optimizer for finding local minima.

    Parameters
    ----------
    learning_rate : float
        The learning rate used for updating the parameters.
    grad_func : callable
        The gradient of the function to be optimized.
    init_guess : float, tuple, or numpy.ndarray
        Initial guess for the minima input.
    batch_size : int
        Number of data points to use in each mini-batch.
    num_mini_batches : int
        Number of mini-batches to use in each iteration.

    """

    def __init__(self, learning_rate: float, grad_func: callable,
                 init_guess: float | np.ndarray | tuple, batch_size: int,
                 num_mini_batches: int):
        super().__init__(learning_rate, grad_func, init_guess)
        self.batch_size = batch_size
        self.num_mini_batches = num_mini_batches

    def advance(self, guess):
        """
        Advance minima value guess using Stochastic Gradient Descent with mini-batches.

        Parameters
        ----------
        guess : float, tuple, or numpy.ndarray
            Current guess for the minima.

        Returns
        -------
        float, tuple, or numpy.ndarray
            Updated minima value guess.

        """
        gradient_sum = np.zeros_like(guess)

        for _ in range(self.num_mini_batches):
            mini_batch_indices = np.random.choice(
                len(guess), size=self.batch_size, replace=False)
            mini_batch = guess[mini_batch_indices]
            gradient_sum += self.grad_func(mini_batch)

        average_gradient = gradient_sum / self.num_mini_batches
        return guess - self.learning_rate * average_gradient

    def __call__(self, iterations_max: int, tolerance: float):
        """
        Compute the minima estimate using Stochastic Gradient Descent.

        Parameters
        ----------
        iterations_max : int
            Maximum number of iterations.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        float, tuple, or numpy.ndarray
            Estimated minima.
        int
            Number of iterations performed.

        """
        if not isinstance(iterations_max, int):
            raise TypeError(
                f"iterations_max: expected int, got {type(iterations_max)}")
        if not isinstance(tolerance, float):
            raise TypeError(
                f"tolerance: expected float, got {type(tolerance)}")
        iteration = 0
        guess = self.init_guess
        while True:
            new_guess = self.advance(guess)
            change = new_guess - guess
            iteration += 1
            guess = new_guess
            if iteration >= iterations_max or np.all(np.abs(change) < tolerance):
                break
        n_iterations = iteration
        return guess, n_iterations

def test_stochastic_gradient_descent():
    optimizer = StochasticGradientDescent()
    def simple_polynomial(a,b,c,d,x):
        return a*x**3 + b*x**2 + c*x + d
    
    def meta_mse():
        model = simple_polynomial(a,b,c,d,x)
        