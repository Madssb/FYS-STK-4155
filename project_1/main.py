"""
Solve project here
"""
import numpy as np
from utilities import (franke_function, mean_squared_error, r2_score, my_figsize)
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
sns.set_theme()

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
  noise = np.random.normal(0, 1, x_mesh.shape)
  mock_data = analytic + 0.1*noise
  mock_data = mock_data - np.mean(mock_data)
  degrees = np.arange(1,6)
  hyperparameters = np.logspace(-4,0,10)
  linreg_instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters,
                                       center=True)
 
  # Task a
  linreg_instance.visualize_mse_ols(show=False, save=True)
  linreg_instance.visualize_r2_ols(show=False, save=True)
  print(linreg_instance.mses_ols)
  print(linreg_instance.r2s_ols)
  # Task b
  linreg_instance.visualize_mse_ridge(show=False, save=True, cbarmin=0, cbarmax=0.12)
  # Task c
  linreg_instance.visualize_mse_lasso(show=False, save=True, cbarmin=0., cbarmax=0.12)
  # Task e
  linreg_instance.visualize_mse_train_test_ols(show=False, save=True)

  mses = np.empty(len(degrees))
  biases = np.empty(len(degrees))
  variances = np.empty(len(degrees))
  for j in range(len(degrees)):
    mse, bias, variance = linreg_instance.bootstrap(nbootstraps=100, 
      degree=degrees[j])
    mses[j] = mse
    biases[j] = bias
    variances[j] = variance
  
  fig, ax = plt.subplots(figsize=my_figsize())
  ax.plot(degrees, mses, label='MSE')
  ax.plot(degrees, biases, label='Bias')
  ax.plot(degrees, variances, label='Variance')
  ax.set_xlabel("Polynomial degree")
  ax.set_ylabel("Error")
  ax.legend()
  fig.tight_layout()
  fig.savefig('../plots/bias_variance.pdf')
  #plt.show()
  
  # Task f
  k = np.arange(5, 11)
  mses_cv = np.empty((len(k), len(degrees)))
  fig, ax = plt.subplots(figsize=my_figsize())
  for i in range(len(k)):
    for j in range(len(degrees)):
      mse, r2 = linreg_instance.cross_validation(k=k[i], 
        degree=degrees[j], method='ols')
      mses_cv[i,j] = mse
    ax.plot(degrees, mses_cv[i,:],label='k='+str(k[i]))
  #ax.plot(degrees, mses, '--', label='Bootstrap MSE')
  #ax.plot(degrees, biases, '--', label='Bias')
  #ax.plot(degrees, variances, '--', label='Variance')
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

  # Put a legend to the right of the current axis
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  #ax.legend(ncols=2)
  ax.set_xlabel("Polynomial degree")
  ax.set_ylabel("MSE")
  fig.tight_layout()
  fig.savefig('../plots/cross_validation.pdf')
  #plt.show()
  
"""
  # Task g
  from imageio import imread
  terrain1 = imread('../astridbg/SRTM_data_Norway_1.tif')
  x_pos, y_pos = 500, 500
  reduce_factor = 30
  y_shift = 1000
  x_shift = 1000
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

  fig.savefig("../plots/terrain_visualization.pdf")
  plt.show()

  x = np.arange(np.shape(z)[0])
  y = np.arange(np.shape(z)[1])
  z = z - np.mean(z)
  z = z / np.std(z)
  degrees = np.linspace(1,15,15,dtype=int)
  hyperparameters = np.logspace(-4,0,10)
  linreg_instance = LinearRegression2D(x, y, z,
                                       degrees, hyperparameters, 
                                       center=True, standardize=True)

  #linreg_instance.visualize_mse_ols(show=False, save=True)
  k = np.linspace(5, 10, 6, dtype=int)
  mses_cv = np.empty((len(k), len(degrees)))
  fig, ax = plt.subplots(figsize=my_figsize())
  for i in range(len(k)):
    for j in range(len(degrees)):
      mse, r2 = linreg_instance.cross_validation(k=k[i], 
        degree=degrees[j], method='ols')
      mses_cv[i,j] = mse
    ax.plot(degrees, mses_cv[i,:],label=str(k[i])+'-fold')
  ax.legend(ncols=2)
  fig.tight_layout()
  fig.savefig('../plots/terrain_cross_val_ols')
  plt.show()

  #linreg_instance.visualize_mse_ridge(show=False, save=True)
  k = 5
  mses_cv = np.empty((len(degrees), len(hyperparameters)))
  for i in range(len(degrees)): 
    for j in range(len(hyperparameters)):
      mse, r2 = linreg_instance.cross_validation(k=k, 
        degree=degrees[i], method='ridge', hyperparameter=hyperparameters[j])
      mses_cv[i,j] = mse
  print(mses_cv)
  fig, ax = plt.subplots(figsize=my_figsize())
  degrees_mesh, hyperparameters_mesh = np.meshgrid(
        degrees, hyperparameters)
  levels = np.linspace(mses_cv.min(), mses_cv.max(), 7)
  contour = ax.contourf(
        degrees_mesh, hyperparameters_mesh, mses_cv.T, levels=levels)
  ax.set_yscale("log")
  ax.set_xlabel("Polynomial degree")
  ax.set_ylabel(r"hyperparameter $\lambda$")
  def format_func(x, _): return f"{x:.2f}"
  cbar = plt.colorbar(contour, format=format_func)
  cbar.set_label('MSE')
  fig.tight_layout()
  plt.show()
  fig.savefig("../plots/terrain_ridge_mse.pdf")

  #linreg_instance.visualize_mse_lasso(show=True, save=False)
  k = 5
  mses_cv = np.empty((len(degrees), len(hyperparameters)))
  for i in range(len(degrees)): 
    for j in range(len(hyperparameters)):
      mse, r2 = linreg_instance.cross_validation(k=k, 
        degree=degrees[i], method='lasso', hyperparameter=hyperparameters[j])
      mses_cv[i,j] = mse
  print(mses_cv)
  fig, ax = plt.subplots(figsize=my_figsize())
  degrees_mesh, hyperparameters_mesh = np.meshgrid(
        degrees, hyperparameters)
  levels = np.linspace(mses_cv.min(), mses_cv.max(), 7)
  contour = ax.contourf(
        degrees_mesh, hyperparameters_mesh, mses_cv.T, levels=levels)
  ax.set_yscale("log")
  ax.set_xlabel("Polynomial degree")
  ax.set_ylabel(r"hyperparameter $\lambda$")
  def format_func(x, _): return f"{x:.2f}"
  cbar = plt.colorbar(contour, format=format_func)
  cbar.set_label('MSE')
  fig.tight_layout()
  plt.show()
  fig.savefig("../plots/terrain_lasso_mse.pdf")                                   
"""
if __name__ == '__main__':
  #ols_franke_function()temporary_name
  #ridge_franke_function()
  main() 