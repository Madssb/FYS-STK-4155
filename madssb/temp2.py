
"""
using autograd in multiple dimensions
"""
from jax import grad
import jax.numpy as np

# Define the objective function of four variables
def f(a, b, c, d):
    return a**2 + b**2 + c**2 + d**2

# Initial guesses for a, b, c, and d
a_init = 4.0
b_init = 3.0
c_init = 2.0
d_init = 1.0

learning_rate = 0.01
iterations_max = 1000
tolerance = 1e-5

# Initial guesses
a_guess = a_init
b_guess = b_init
c_guess = c_init
d_guess = d_init
iteration = 0

# Lists to store the trajectory of guesses
a_guesses = [a_guess]
b_guesses = [b_guess]
c_guesses = [c_guess]
d_guesses = [d_guess]

while True:
    grad_f = grad(f, argnums=(0, 1, 2, 3))(a_guess, b_guess, c_guess, d_guess)

    new_a_guess = a_guess - learning_rate * grad_f[0]
    new_b_guess = b_guess - learning_rate * grad_f[1]
    new_c_guess = c_guess - learning_rate * grad_f[2]
    new_d_guess = d_guess - learning_rate * grad_f[3]

    a_guesses.append(new_a_guess)
    b_guesses.append(new_b_guess)
    c_guesses.append(new_c_guess)
    d_guesses.append(new_d_guess)

    change_a = new_a_guess - a_guess
    change_b = new_b_guess - b_guess
    change_c = new_c_guess - c_guess
    change_d = new_d_guess - d_guess

    a_guess = new_a_guess
    b_guess = new_b_guess
    c_guess = new_c_guess
    d_guess = new_d_guess

    iteration += 1

    if iterations_max <= iteration or np.sqrt(change_a**2 + change_b**2 + change_c**2 + change_d**2) < tolerance:
        break
print(a_guess)
print(b_guess)
print(c_guess)
print(d_guess)