"""
Solve project here
"""
import numpy as np
from utilities import (franke_function, mean_squared_error, r2_score, my_figsize)
from sklearn.model_selection import train_test_split
import pandas as pd

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from regression import LinearRegression2D


def main():
  """
  Generate random input, and Franke function data, compute regression
  for OLS, Ridge and Lasso, evaluate MSE and R2, and visualize.
  
  
  """
  np.random.seed(2023)
  x = np.linspace(0, 1, 40)
  y = np.linspace(0, 1, 40) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 0.1, x_mesh.shape)
  mock_data = analytic + noise
  degrees = np.linspace(1,20,20,dtype=int)
  hyperparameters = np.logspace(-4,4,10)
  linreg_instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters)
  # Task a
  #linreg_instance.visualize_mse_ols(show=True, save=False)
  # Task b
  #linreg_instance.visualize_mse_ridge(show=True, save=False)
  # Task c
  #linreg_instance.visualize_mse_lasso(show=True, save=False)
  # Task e
  print(linreg_instance.bootstrap(nbootstraps=100, degree=3))

  # Task f
  k = [5, 6, 7, 8, 9, 10]
  mses_cv = np.empty((len(k), len(degrees)))
  plt.figure()
  for i in range(len(k)):
    for j in range(len(degrees)):
      mse, r2 = linreg_instance.cross_validation(k=k[i], 
        degree=degrees[j], method='ols')
      mses_cv[i,j] = mse
    plt.plot(degrees, mses_cv[i,:],label=k[i])
  plt.legend()
  plt.show()
  print(mses_cv)

  

  # Task g
  
  from imageio import imread
  #import matplotlib.pyplot as plt
  #from mpl_toolkits.mplot3d import Axes3D
  #from matplotlib import cm
  """
  terrain1 = imread('../astridbg/SRTM_data_Norway_1.tif')
  z = terrain1[::30, ::30]
  x = np.arange(np.shape(z)[0])
  y = np.arange(np.shape(z)[1])
  print(len(x))
  print(len(y))
  print(np.shape(z))
  degrees = np.linspace(1,12,12,dtype=int)
  hyperparameters = np.logspace(-4,4,10)
  linreg_instance = LinearRegression2D(x, y, z,
                                       degrees, hyperparameters, 
                                       center=True, normalize=True)
  linreg_instance.visualize_mse_ols(show=True, save=False)
     """                                  


if __name__ == '__main__':
  #ols_franke_function()temporary_name
  #ridge_franke_function()
  main() 