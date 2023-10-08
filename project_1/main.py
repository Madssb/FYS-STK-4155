"""
Solve project here
"""
import numpy as np
from utilities import (franke_function, my_figsize)
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
  x = np.arange(0, 1, 0.05)
  y = np.arange(0, 1, 0.05) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  mock_data = analytic + noise
  degrees = np.linspace(1,5,5,dtype=int)
  hyperparameters = np.logspace(-4,4,10, dtype=float)
  linreg_instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters)
  linreg_instance.visualize_quantities_ols()



if __name__ == '__main__':
  main() 