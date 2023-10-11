"""
File exists as attempted solution to week 41 FYS-STK exercices.
"""
import numpy as np
import matplotlib.pyplot as plt


def simple_func(x: np.ndarray | float):
  """
  simple polynomial
  """
  return 1 + 2*x + 3*x**2


def simple_func_diff(x):
  return 2 + 6*x


class GradientDescentSimpleCase:
  """
  Eh
  """
  def __init__(self, input: np.ndarray, output: np.ndarray,
               learning_rate: float, momentum: float, iterations_max: int=1000,
               tol: float=1e-5):
    """
    Instantiate GradientDescentSimpleCasse object
    """
    self.input = input
    self.output = output
    self.extremal_guess = extremal_guess
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.iterations_max = iterations_max
    self.tol = tol
    self.extremal_guesses = {}

  def gradient_descent(self, extremal_guess):
    """
    Minimize simple func using gradient descent.
    """
    iteration = 0
    change = 1
    extremal_estimate = extremal_guess
    extremal_estimates = np.empty(self.iterations_max, dtype=float)
    while self.iterations_max > iteration and (np.abs(change) > self.tol):
      extremal_estimates[iteration] = extremal_estimate
      change = self.learning_rate * simple_func_diff(extremal_estimate)
      extremal_estimate -=  change
      iteration += 1
    return extremal_estimates[:iteration], iteration

  def gradient_descent_with_momentum(self, extremal_guess):
    """
    Minimize simple func using gradient descent with momentum.
    """
    iteration = 0
    new_change = 1
    change = 0
    extremal_estimate = extremal_guess
    extremal_estimates = np.empty(self.iterations_max, dtype=float)
    while self.iterations_max > iteration and (np.abs(new_change) > self.tol):
      extremal_estimates[iteration] = extremal_estimate
      new_change = self.learning_rate * simple_func_diff(extremal_estimate)\
        + self.momentum*change
      change = new_change
      extremal_estimate -=  change
      iteration += 1
    return extremal_estimates[:iteration], iteration

def stochastic_gradient_descent(self, n_batches):
  """
  Minimize simple func using stochastic gradient descent with momentum.
  """
  batches = np.array_split(input, n_batches)
  extremal_val_candidates = np.empty(n_batches)
  for i, batch in enumerate(batches):
    extremal_guess = np.random.choice(batch)
    extremal_val_candidates[i] = self.gradient_descent(extremal_guess)[-1]
  extremal_val = extremal_val_candidates[0]
  for extremal_val_candidate in extremal_val_candidates:
    if simple_func(extremal_val_candidate) > simple_func(extremal_val):
      extremal_val_best_estimate = extremal_val_candidate
  return extremal_val_best_estimate




if __name__ == "__main__":
  input = np.linspace(-10,10,100, dtype=float)
  output = simple_func(input)
  instance = GradientDescentSimpleCase(input, output, 1e-2, 0.5)
  extremal_guess = input[80]
  extremal_estimates_gd, n_iterations_gd = instance.gradient_descent(extremal_guess)
  extremal_estimates_adam, n_iterations_adam = \
    instance.gradient_descent_with_momentum(extremal_guess)
  extremal_guess_sgd = instance.stochastic_gradient_descent(5)
  print(f"{n_iterations_gd=}")
  print(f"{n_iterations_adam=}")
  plt.plot(extremal_estimates_gd, simple_func(extremal_estimates_gd), "rx",
           label="GD")
  plt.plot(extremal_estimates_adam, simple_func(extremal_estimates_adam), "bx",
           label="MGD")
  plt.plot(input, output, label ="func")
  plt.legend()
  plt.show()





# class GradientDescent:
#   def __init__(self, input, output, max_degree, parameters_guess, momentum,
#                iterations_max=10000, tolerance=1e-5):
#     assert len(parameters_guess.shape) == 1
#     assert parameters_guess.shape[0] == max_degree + 1
#     self.input = input
#     self.n_features = max_degree + 1
#     self.features = np.column_stack(
#         [input ** i for i in range(self.n_features)])
#     self.output = output
#     self.iterations_max = iterations_max
#     self.tolerance = tolerance
#     self.n = input.shape[0]
#     #self.learning_rate = 2/self.n * self.features.T @ self.features
#     self.learning_rate = 1e-1
#     self.parameters_init = parameters_guess.astype(float)
#     self.momentum = momentum

#   def gradient(self, parameters):
#     """
#     Compute gradient for cost function w.r.t parameters
#     """
#     gradient = 2/self.n * self.features.T @ (
#         self.features @ parameters - self.output)
#     return gradient

#   def gradient_descent(self, print_performance= True):
#     """
#     Compute optimal parameters with gradient descent.
#     """
#     iteration = 0
#     change = 1
#     parameters = self.parameters_init
#     while self.iterations_max > iteration and \
#       (np.abs(change) > self.tolerance).any():
#       change = self.learning_rate * self.gradient(parameters)
#       parameters -=  change
#       iteration += 1
#     if print_performance:
#       print(f"""number of iterations: {iteration}
#       converged = {(change < self.tolerance).all()}""")
#     return parameters

#   def gradient_descent_with_momentum(self, print_performance=True):
#     """
#     Compute optimal parameters with gradient descent with momentum.
#     """
#     iteration = 0
#     change_new = 1
#     change = 0
#     parameters = self.parameters_init
#     while self.iterations_max > iteration and \
#       (np.abs(change_new) > self.tolerance).any():
#       gradient = self.gradient(parameters)
#       change_new = self.learning_rate * gradient + self.momentum*change
#       change = change_new
#       parameters -= change_new
#       iteration += 1
#     if print_performance:
#       print(f"""number of iterations: {iteration}
#       converged = {(change_new < self.tolerance).all()}""")
#     return parameters

#   def __call__(self, method):
#     methods = [self.gradient_descent_with_momentum, self.gradient_descent]
#     assert method in methods
#     print(f"{method.__name__} Performance:")
#     predicted = self.features @ method()
#     return predicted


# if __name__ == "__main__":
#   input = np.linspace(-1, 1, 100, dtype=float)
#   output = simple_func(input)
#   instance = GradientDescent(input, output, 2, np.array([1, 1, 1]), 0.5)
#   predicted_adam = instance(instance.gradient_descent_with_momentum)
#   predicted_gd = instance(instance.gradient_descent)
#   #plt.plot(input, predicted_adam, "bx", label="ADAM")
#   plt.plot(input, predicted_gd, "rx", label="GD")
#   plt.plot(input, output, label="output")
#   plt.xlabel("x")
#   plt.ylabel("y")
#   plt.legend()
#   plt.show()

