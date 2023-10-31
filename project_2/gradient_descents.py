
import numpy as np
class GradientDescent:
  """
  Find local minima with Gradient Descent
  """
  def __init__(self, learning_rate: float, grad_func: callable, 
               init_guess: float | np.ndarray | tuple, *additional_grad_args):
    """
    Initialize GradientDescent


    Parameters
    ----------
    learning_rate: float
      Learning rate.
    grad_func: callable
      gradient of func to be optimized
    init_guess: float
      initial guess for minima input
    """
    self.learning_rate = learning_rate
    self.grad_func = grad_func
    self.init_guess = init_guess
    self.additional_grad_args = additional_grad_args
  
  def advance(self, guess):
    """
    Advance minima value guess
    """
    return guess - self.learning_rate*self.grad_func(guess, *self.additional_grad_args)
  
  def __call__(self, iterations_max: int, tolerance: float):
    """
    Compute minima
    """
    iteration = 0
    guess = self.init_guess
    guesses = [guess]
    while True:
      new_guess = self.advance(guess)
      change = new_guess - guess
      iteration += 1
      guess = new_guess
      if iteration >= iterations_max or np.abs(change).all() < tolerance:
        break
    iterations_max_exceeded = iterations_max <= iteration
    tolerance_met = np.abs(change) < tolerance
    return guess, iterations_max_exceeded, tolerance_met