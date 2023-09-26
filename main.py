"""
Solve project here
"""
import numpy as np
from utilities import (franke_function, mean_squared_error, r2_score, my_figsize)
from regression import (ols_regression, features_polynomial_xy, ridge_regression)
from sklearn.model_selection import train_test_split
import pandas as pd

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from regression import LinearRegression2D


def ols_franke_function():
  """
  Create a mesh for the franke function, and convert aforementioned mesh to
  some mock data by adding gaussian noise. Next, split data into trainining set
  and test set, and train for optimal parameters with OLS using training set,
  for complexities spanning polynomial order one to five. 
  Then evaluate models with MSE and R2, and visualize aforementioned quantities
  as function of the complexity.


  Returns
  -------
  None


  """
  # Make data.
  x = np.arange(0, 1, 0.05)
  y = np.arange(0, 1, 0.05)
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  polynomial_degrees = np.linspace(1, 5, 5, dtype=int)
  mses = np.empty_like(polynomial_degrees, dtype=float)
  r2s = np.empty_like(polynomial_degrees, dtype=float)
  mock_data = analytic + noise
  data_flat = mock_data.ravel()
  for idx, polynomial_degree in enumerate(polynomial_degrees):
    features = features_polynomial_xy(
        x, y, polynomial_degree, scale=False)
    features_train, features_test, data_train, data_test = (
      train_test_split(features, data_flat))
    ols_parameters = ols_regression(features_train, data_train)
    ols_model_predicted = features_test @ ols_parameters
    mses[idx] = mean_squared_error(data_test, ols_model_predicted)
    r2s[idx] = r2_score(data_test, ols_model_predicted)
    print(
        f"degree {polynomial_degree}: mse = {mses[idx]:.4g}, r2 = {r2s[idx]:.4g}")
  plt.plot(polynomial_degrees, mses, label="MSE")
  plt.plot(polynomial_degrees, r2s, label="R2")

  plt.legend()
  plt.show()


def ridge_franke_function():
  """
  Create a mesh for the franke function, and convert aforementioned mesh to
  some mock data by adding gaussian noise. Next, split data into trainining set
  and test set, and train for optimal parameters with using regression, for
  complexities spanning one to five, and for various hyperparameters. Then
  evaluate models with MSE and r2.
  """
  x = np.arange(0, 1, 0.05)
  y = np.arange(0, 1, 0.05)
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  mock_data = analytic + noise
  data_flat = mock_data.ravel()
  polynomial_degrees = np.linspace(1, 5, 5, dtype=int)
  #hyperparams = np.linspace(10**-4,10**4,10)
  hyperparams = np.logspace(-4,4,10)
  mse = np.empty((polynomial_degrees.shape[0],hyperparams.shape[0]),dtype=float)
  r2s = np.empty_like(mse, dtype=float)
  for pol_idx, polynomial_degree in enumerate(polynomial_degrees):
    features = features_polynomial_xy(
        x, y, polynomial_degree, scale=False)
    features_train, features_test, data_train, data_test = (
      train_test_split(features, data_flat))
    for hyp_idx, hyperparam in enumerate(hyperparams):
      ridge_parameters = ridge_regression(
        features_train,data_train, hyperparam)
      rigdge_model_predicted = features_test @ ridge_parameters
      mse[pol_idx,hyp_idx] = mean_squared_error(data_test, rigdge_model_predicted)
      r2s[pol_idx, hyp_idx] = r2_score(data_test, rigdge_model_predicted)
  
  fig, ax = plt.subplots(figsize=my_figsize())
  polynomial_degrees_mesh, hyperparams_mesh = np.meshgrid(polynomial_degrees, hyperparams)
  levels = np.linspace(mse.min(),mse.max(), 7)
  contour = ax.contourf(polynomial_degrees_mesh, hyperparams_mesh, mse.T, levels=levels)
  ax.set_yscale("log")
  ax.set_xlabel("complexity")
  ax.set_ylabel("Ridge parameter")
  ax.grid()
  format_func = lambda x, _: f"{x:.2f}"
  cbar = plt.colorbar(contour, format=format_func)
  fig.tight_layout()
  fig.savefig("ridge_mse.pdf")




def main():
  x = np.arange(0, 1, 0.05)
  y = np.arange(0, 1, 0.05) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  mock_data = analytic + noise
  degrees = np.linspace(1,5,5,dtype=int)
  hyperparameters = np.logspace(-4,4,10)
  linreg_instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters)
  linreg_instance.mse_ols()
if __name__ == '__main__':
  ols_franke_function()
  ridge_franke_function()
  #main()