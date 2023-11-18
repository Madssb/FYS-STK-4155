"""
apply sklearn neural network with regression
"""
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error


def regression_1d_poly_2_deg():
# Set a random seed for reproducibility
  np.random.seed(2024)
  rng = np.random.default_rng(2023)
  n = 100
  x = np.linspace(-10, 10, n)
  y = 4 + 3 * x + x**2 + rng.normal(0, 0.1, n)

  X = np.array([np.ones(100), x, x**2]).T
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split the data into train and test sets

  # Create and train the MLPRegressor
  model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)
  model.fit(X_train, y_train)

  # Make predictions on the test set
  predicted = model.predict(X_test)

  # Calculate mean squared error
  mse = mean_squared_error(y_test, predicted)
  print("Mean Squared Error:", mse)


regression_1d_poly_2_deg()