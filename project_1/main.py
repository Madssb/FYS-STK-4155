"""
Solve project here
"""
import numpy as np
from utilities import (franke_function, convert_to_label, my_figsize)
from model_evaluation_metrics import (mean_squared_error, r2_score, bias,
                                      variance)
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from regression import LinearRegression2D
import warnings
from sklearn.exceptions import ConvergenceWarning


def make_figs_for_everything(instance: LinearRegression2D, data: np.ndarray,
                             data_str: str):
  """
  Visualize all.
  """
  model_eval_funcs = [mean_squared_error, r2_score, bias, variance]
  eval_funcs_str = ["mse", "r2", "bias", "variance"]
  regression_methods = [instance.ols, instance.ridge, instance.lasso]
  regression_methods_str = ["ols", "ridge", "lasso"]

  eval_prediction_methods = [instance.evaluate_predicted,
                             instance.evaluate_predicted_crossval]
  crossval_str = ["", "cross_val"]
  n_pts = str(data.shape[0])
  for i, model_eval_func  in enumerate(model_eval_funcs):
    for j, regression_method in enumerate(regression_methods):
      for k, eval_prediction_method in enumerate(eval_prediction_methods):
        evaled_model_mesh = instance.evaluate_model_mesh(regression_method,
                                                         model_eval_func,
                                                         eval_prediction_method)
        if regression_method == instance.ols:
          fig, ax = instance.visualize_ols(evaled_model_mesh, eval_funcs_str[i].upper())
          filename = f"figs/{data_str}_{eval_funcs_str[i]}_{regression_methods_str[j]}_{crossval_str[k]}_{n_pts}.pdf"
          fig.savefig(filename)
        else:
          fig, ax = instance.visualize_mse_ridge(evaled_model_mesh, eval_funcs_str[i].upper())
          filename = f"figs/{data_str}_{eval_funcs_str[i]}_{regression_methods_str[j]}_{crossval_str[k]}_{n_pts}.pdf"
          fig.savefig(filename)



def simple_degree_analysis():
  """
  Compute predicted for franke function mesh with synthetic noise, with ols and
  complexity ranging from 1 degree to 5 degree order x and y polynomial.
  Visualize aforementioned predictions, and franke function mehs, with and
  without synthetic noise.
  """
  np.random.seed(2023)
  points = 20
  sigma = 0.1
  x = np.arange(0, 1, 1/points)
  y = np.arange(0, 1, 1/points) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, sigma, x_mesh.shape)
  mock_data = (analytic + noise).ravel()
  degrees = np.arange(1, 6, dtype=int)

  instance = LinearRegression2D(x, y, mock_data)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(x_mesh, y_mesh,mock_data.reshape(x_mesh.shape),
                  cmap='viridis')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  fig.savefig("figs/franke_function_w_noise.pdf")
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(x_mesh, y_mesh,analytic, cmap='viridis')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  fig.savefig("figs/franke_function_wo_noise.pdf")
  for degree in degrees:
    features = instance.features_polynomial_xy(degree)
    predicted = instance.ols(features, features, mock_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    ax.plot_surface(x_mesh, y_mesh, predicted.reshape(x_mesh.shape),
                    cmap='viridis')
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.savefig(f"figs/franke_function_predicted_{degree}_degrees.pdf")


def franke_simple_mse_and_r2_analysis():
  """
  Compute predicted for franke function mesh with synthetic noise, with OLS,
  Ridge and Lasso regression for complexities spanning one to five degrees,
  and hyperparameter logspace of 10**-4 to 10**4. Evaluate predicted with
  MSE and r2, and visualize model evaluation.
 

  """
  np.random.seed(2023)
  x = np.arange(0, 1, 0.05)
  y = np.arange(0, 1, 0.05) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  mock_data = (analytic + noise).ravel()
  degrees = np.arange(1, 6, dtype=int)
  hyperparameters = np.logspace(-4,4,10, dtype=float)
  instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters)  
  regression_methods = [instance.ols, instance.ridge, instance.lasso]
  eval_funcs = [mean_squared_error, r2_score]
  for regression_method in regression_methods:
    for eval_func in eval_funcs:
      eval_model_mesh = \
          instance.evaluate_model_mesh(regression_method,
                                       eval_func,
                                       instance.evaluate_predicted)
      suffix = f"{regression_method.__name__}_{eval_func.__name__}.pdf"
      filename = f"figs/simple_franke_{suffix}"
      ylabel = convert_to_label(eval_func.__name__)
      if regression_method == instance.ols:
        fig, ax = instance.visualize_ols(eval_model_mesh, ylabel)
        fig.savefig(filename)
      else:
        fig, ax = instance.visualize_mse_ridge(eval_model_mesh, ylabel)
        fig.savefig(filename)


def cross_validation_analysis():
  """
  
  """
  np.random.seed(2023)
  x = np.arange(0, 1, 0.05)
  y = np.arange(0, 1, 0.05) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  mock_data = (analytic + noise).ravel()
  degrees = np.arange(1, 6, dtype=int)
  hyperparameters = np.logspace(-4,4,10, dtype=float)
  instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters) 
  k_folds = np.arange(5,11, dtype=int)
  mean_mses = np.empty_like(k_folds, dtype=float)
  fig, ax = plt.subplots(figsize=my_figsize())
  regression_methods = [instance.ols, instance.ridge, instance.lasso]
  for regression_method in regression_methods:
    for i, k_fold in enumerate(k_folds):
      mses = instance.evaluate_model_mesh(regression_method,
                                          mean_squared_error,
                                          instance.evaluate_predicted_crossval,
                                          k_fold)
      mean_mses[i] = np.mean(mses)
    label = convert_to_label(regression_method.__name__)
    ax.plot(k_folds, mean_mses, label=label)
  ax.set_xlabel("k-folds")
  ax.set_ylabel("mean MSE")
  fig.legend()
  fig.tight_layout()
  fig.savefig(f"figs/crossval_analysis_mse_.pdf")


def terrain():
  """
  TBA
  """
  img = Image.open("data/SRTM_data_Norway_1.tif")
  data = np.array(img)
  n_pts = 1000
  ds_factor = int(np.round(np.sqrt((data.shape[0]*data.shape[1])/n_pts)))
  data_downsampled = data[::ds_factor,::ds_factor]
  x = np.linspace(0,1,data_downsampled.shape[0])
  y = np.linspace(0,1,data_downsampled.shape[1])
  z = data_downsampled.ravel().astype(np.float64)
  degrees = np.arange(1,16)
  hyperparameters = np.logspace(-4,0,10, dtype=float)
  instance = LinearRegression2D(x, y, z, degrees, hyperparameters)
  print("made it here")
  mse_crossval = instance.evaluate_model_mesh(
    instance.ols, mean_squared_error, instance.evaluate_predicted_crossval)
  fig, ax = instance.visualize_ols(mse_crossval, "MSE")
  make_figs_for_everything(instance, z, "terrain")


def total_mses_franke():
  """
  compute mean MSEs for OLS, Ridge and Lasso models fitting Frankes function
  output with synthetic noise, cross validated and not.


  """
  np.random.seed(2023)
  x = np.arange(0, 1, 0.05)
  y = np.arange(0, 1, 0.05) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  mock_data = (analytic + noise).ravel()
  degrees = np.arange(1, 6, dtype=int)
  hyperparameters = np.logspace(-4,4,10, dtype=float)
  instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters)
  regression_methods = [instance.ols, instance.ridge, instance.lasso]
  eval_pred_methods = [instance.evaluate_predicted,
                       instance.evaluate_predicted_crossval]
  for eval_pred_method in eval_pred_methods:
    print(eval_pred_method.__name__)
    for regression_method in regression_methods:
      mse = instance.evaluate_model_mesh(regression_method, mean_squared_error,
                                         eval_pred_method)
      mean_mse = np.mean(mse)
      print(f"mean mse {regression_method.__name__}: {mean_mse:.4g}")


def total_mses_terrain():
  """
  Compute mean MSEs for OLS, Ridge and Lasso models fitting terrain data,
  cross validated and not.


  """
  img = Image.open("data/SRTM_data_Norway_1.tif")
  data = np.array(img)
  n_pts = 1000
  ds_factor = int(np.round(np.sqrt((data.shape[0]*data.shape[1])/n_pts)))
  data_downsampled = data[::ds_factor,::ds_factor]
  x = np.linspace(0,1,data_downsampled.shape[0])
  y = np.linspace(0,1,data_downsampled.shape[1])
  z = data_downsampled.ravel().astype(np.float64)
  degrees = np.arange(1,16)
  hyperparameters = np.logspace(-4,0,10, dtype=float)
  instance = LinearRegression2D(x, y, z, degrees, hyperparameters)
  regression_methods = [instance.ols, instance.ridge, instance.lasso]
  eval_pred_methods = [instance.evaluate_predicted,
                       instance.evaluate_predicted_crossval]
  print("terrain, {z.shape[0]} datapoints")
  for eval_pred_method in eval_pred_methods:
    print(eval_pred_method.__name__)
    for regression_method in regression_methods:
      mse = instance.evaluate_model_mesh(regression_method, mean_squared_error,
                                         eval_pred_method)
      mean_mse = np.mean(mse)
      print(f"mean mse {regression_method.__name__}: {mean_mse:.4g}")

if __name__ == '__main__':
  warnings.filterwarnings('ignore', category=ConvergenceWarning)
  #simple_degree_analysis()
  #franke_simple_mse_and_r2_analysis()
  cross_validation_analysis()
  #franke()
  #terrain()
  #total_mses_franke()
  #total_mses_terrain()