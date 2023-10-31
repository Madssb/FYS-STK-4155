"""
grad function in JAX implemented for computing four-dimensional gradient!
"""
from jax import grad
import jax.numpy as np

# Define the objective function of four variables
def f(a, b, c, d, x):
    return a**2 + b**2 + c**2 + d**2 + x

# Initial guesses for a, b, c, and d
a_init = 4.0
b_init = 3.0
c_init = 2.0
d_init = 1.0

init_array = np.array([a_init, b_init, c_init, d_init])

learning_rate = 0.01
iterations_max = 1000
tolerance = 1e-5

# Initial guesses
guess_array = init_array
iteration = 0

# Lists to store the trajectory of guesses
guesses_array = [init_array]
x = 2

while True:
    n_parameters = guess_array.shape[0]
    param_indices_tuple = tuple(np.arange(n_parameters))
    grad_f = grad(f, argnums=param_indices_tuple)(*guess_array, x)  # Unpack the array using *

    # Element-wise operations with NumPy arrays
    new_guess_array = guess_array - learning_rate * np.array(grad_f)
    guesses_array.append(new_guess_array)
    change_array = new_guess_array - guess_array

    guess_array = new_guess_array
    iteration += 1

    if iterations_max <= iteration or np.sqrt(np.sum(change_array**2)) < tolerance:
        break

print(guesses_array[-1])
