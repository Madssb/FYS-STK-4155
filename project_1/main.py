"""
Solve project here
"""
import numpy as np
from utilities import (franke_function, my_figsize)
from sklearn.model_selection import train_test_split
from model_evaluation_metrics import (mean_squared_error, r2_score, bias,
                                      variance)
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
  instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters)
  mses_ols = instance.evaluate_model_mesh(instance.ols,
                                                 mean_squared_error,
                                                 instance.evaluate_predicted)
  mses_ridge = instance.evaluate_model_mesh(instance.ridge,
                                                 mean_squared_error,
                                                 instance.evaluate_predicted)
  
  model_eval_funcs = [mean_squared_error, r2_score, bias, variance]
  eval_funcs_str = ["mse", "r2", "bias", "variance"]
  regression_methods = [instance.ols, instance.ridge, instance.lasso]
  regression_methods_str = ["ols", "ridge", "lasso"]

  eval_prediction_methods = [instance.evaluate_predicted,
                             instance.evaluate_predicted_bootstrap]
  bootstrap_str = ["", "bootstrap"]
  n_pts = str(mock_data.ravel().shape[0])
  for i, model_eval_func  in enumerate(model_eval_funcs):
    for j, regression_method in enumerate(regression_methods):
      for k, eval_prediction_method in enumerate(eval_prediction_methods):
        evaled_model_mesh = instance.evaluate_model_mesh(regression_method,
                                                         model_eval_func,
                                                         eval_prediction_method)
        if regression_method == instance.ols:
          fig, ax = instance.visualize_ols(evaled_model_mesh, eval_funcs_str[i])
          filename = f"figs/{eval_funcs_str[i]}_{regression_methods_str[j]}_{bootstrap_str[k]}_{n_pts}.pdf"
          fig.savefig(filename)
        else:
          fig, ax = instance.visualize_mse_ridge(evaled_model_mesh, eval_funcs_str[i])
          filename = f"figs/{eval_funcs_str[i]}_{regression_methods_str[j]}_{bootstrap_str[k]}_{n_pts}.pdf"
          fig.savefig(filename)

if __name__ == '__main__':
  main() 