from jax import grad
import jax.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the objective function of two variables
def f(x, y):
    return x**2 + y**2

# Initial guesses for x and y
x_init = 4.0
y_init = 4.0

learning_rate = 0.01
iterations_max = 1000
tolerance = 1e-5

# Initial guesses
x_guess = x_init
y_guess = y_init
iteration = 0

# Lists to store the trajectory of guesses
x_guesses = [x_guess]
y_guesses = [y_guess]

while True:
    grad_f_x = grad(f, argnums=0)(x_guess, y_guess)
    grad_f_y = grad(f, argnums=1)(x_guess, y_guess)

    new_x_guess = x_guess - learning_rate * grad_f_x
    new_y_guess = y_guess - learning_rate * grad_f_y

    x_guesses.append(new_x_guess)
    y_guesses.append(new_y_guess)

    change_x = new_x_guess - x_guess
    change_y = new_y_guess - y_guess

    x_guess = new_x_guess
    y_guess = new_y_guess

    iteration += 1

    if iterations_max <= iteration or np.sqrt(change_x**2 + change_y**2) < tolerance:
        break

x_guesses_arr = np.array(x_guesses)
y_guesses_arr = np.array(y_guesses)
eval = f(x_guesses_arr, y_guesses_arr)

# Create a 3D surface plot of the objective function
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Plot the trajectory of guesses on the surface plot
ax.scatter(x_guesses_arr, y_guesses_arr, eval, color='black', marker='.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Gradient Descent in Two Dimensions')

plt.show()
