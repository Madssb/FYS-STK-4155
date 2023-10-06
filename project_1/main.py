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
  x = np.linspace(0, 1, 20)
  y = np.linspace(0, 1, 20) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 0.1, x_mesh.shape)
  mock_data = analytic + noise
  mock_data = mock_data - np.mean(mock_data)
  mock_data = mock_data / np.std(mock_data)
  degrees = np.linspace(1,11,11,dtype=int)
  hyperparameters = np.logspace(-4,4,10)
  linreg_instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters,
                                       center=True, normalize=True)
  # Task a
  #linreg_instance.visualize_mse_ols(show=True, save=False)
  # Task b
  #linreg_instance.visualize_mse_ridge(show=True, save=False)
  # Task c
  #linreg_instance.visualize_mse_lasso(show=True, save=False)
  # Task e
 
  mses = np.empty(len(degrees))
  biases = np.empty(len(degrees))
  variances = np.empty(len(degrees))
  for j in range(len(degrees)):
    mse, bias, variance = linreg_instance.bootstrap(nbootstraps=100, 
      degree=degrees[j])
    mses[j] = mse
    biases[j] = bias
    variances[j] = variance
  
  plt.figure()
  plt.plot(degrees, mses, label='mse')
  plt.plot(degrees, biases, label='bias')
  plt.plot(degrees, variances, label='variance')
  plt.legend()
  plt.show()

  # Task f
  k = np.linspace(5, 10, 6, dtype=int)
  mses_cv = np.empty((len(k), len(degrees)))
  plt.figure()
  for i in range(len(k)):
    for j in range(len(degrees)):
      mse, r2 = linreg_instance.cross_validation(k=k[i], 
        degree=degrees[j], method='ols')
      mses_cv[i,j] = mse
    plt.plot(degrees, mses_cv[i,:],label=k[i])
  plt.plot(degrees, mses, '--', label='bootstrap mean')
  plt.plot(degrees, biases, '--', label='bootstrap bias')
  plt.plot(degrees, variances, '--', label='bootstrap variance')
  plt.legend()
  plt.show()

  

  # Task g
  from imageio import imread
  terrain1 = imread('../astridbg/SRTM_data_Norway_1.tif')
  x_pos, y_pos = 500, 500
  reduce_factor = 10
  y_shift = 600
  x_shift = 600
  z = terrain1[y_pos:y_pos+y_shift, x_pos:x_pos+x_shift]
  z = z[::reduce_factor, ::reduce_factor]

  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')


  x = np.arange(np.shape(z)[0])
  y = np.arange(np.shape(z)[1])
  x, y = np.meshgrid(y,x)

  # Plot the surface.
  surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, 
                        linewidth=0, antialiased=False)

  # Customize the z axis.
  #ax.set_zlim(-0.10, 1.40)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)

  plt.show()

  x = np.arange(np.shape(z)[0])
  y = np.arange(np.shape(z)[1])
  z = z - np.mean(z)
  z = z / np.std(z)
  degrees = np.linspace(1,15,15,dtype=int)
  hyperparameters = np.logspace(-4,0,10)
  linreg_instance = LinearRegression2D(x, y, z,
                                       degrees, hyperparameters, 
                                       center=True, normalize=True)
  linreg_instance.visualize_mse_ols(show=True, save=False)
  k = np.linspace(5, 10, 6, dtype=int)
  mses_cv = np.empty((len(k), len(degrees)))
  plt.figure()
  for i in range(len(k)):
    for j in range(len(degrees)):
      mse, r2 = linreg_instance.cross_validation(k=k[i], 
        degree=degrees[j], method='ols', hyperparameter=10**(-4))
      mses_cv[i,j] = mse
    plt.plot(degrees, mses_cv[i,:],label=k[i])
  plt.legend()
  plt.show()

  #linreg_instance.visualize_mse_ridge(show=True, save=False)
  #linreg_instance.visualize_mse_lasso(show=True, save=False)
                                      

if __name__ == '__main__':
  #ols_franke_function()temporary_name
  #ridge_franke_function()
  main() 