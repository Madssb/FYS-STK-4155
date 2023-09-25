"""
Solve project here
"""
import numpy as np
from utilities import franke_function
from utilities import mean_squared_error
from utilities import r2_score
from regression import ols_regression
from regression import design_matrix_polynomial_xy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def ols_model_2d(x,y,z,polynomial_degree):
  """
  produce 2d model
  """
  design_matrix = design_matrix_polynomial_xy(x, y, polynomial_degree)
  ols_parameters = ols_regression(design_matrix, z)
  ols_model = design_matrix @ ols_parameters


def main():
  # Make data.
  x = np.arange(0, 1, 0.05)
  y = np.arange(0, 1, 0.05)
  x_mesh, y_mesh = np.meshgrid(x,y)
  z = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  polynomial_degrees = np.array([1,2,3,4,5],dtype=int)
  mses = np.empty_like(polynomial_degrees, dtype=float)
  r2s = np.empty_like(polynomial_degrees, dtype=float)
  noisy_z = z + noise
  for idx, polynomial_degree in enumerate(polynomial_degrees):
    ols_model = ols_model_2d(x, y, noisy_z, polynomial_degree)
    mses[idx] = mean_squared_error(z, ols_model)
    r2s[idx] = r2_score(z, ols_model)
    print(f"degree {polynomial_degree}: mse = {mses[idx]:.4g}, r2 = {r2s[idx]:.4g}")


if __name__ == '__main__':
  main()