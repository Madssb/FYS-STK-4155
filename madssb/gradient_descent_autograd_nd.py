"""

"""

from jax import grad
import jax.numpy as np
import matplotlib.pyplot as plt

x_init = 14
learning_rate = 0.01
iterations_max = 1000
tolerance = 1e-5

#initial guess
guess = 14.0
iteration = 0
guesses = [guess]
while True:
  new_guess = guess - learning_rate*grad(f)(guess)
  change = new_guess - guess
  guesses.append(new_guess)
  guess = new_guess
  iteration  = iteration + 1
  if iterations_max <= iteration or np.abs(change) < tolerance:
    break
guesses_arr = np.array(guesses)
guesses_eval = f(guesses_arr)
domain = np.linspace(-2,14)
eval = f(domain)

plt.plot(domain,eval)
plt.scatter(guesses_arr, guesses_eval, color='black', marker='.')
plt.plot()
plt.show()