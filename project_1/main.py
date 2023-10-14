"""
Solve project here
"""
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import seaborn as sns 
sns.set_theme()

from model_evaluation_metrics import (mean_squared_error, r2_score, bias,
                                      variance, mean_squared_error_bootstrapped)
from utilities import (franke_function, convert_to_label, my_figsize)
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from mpl_toolkits.mplot3d import Axes3D
from regression import LinearRegression2D
from PIL import Image

# formatting plots and figures
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'axes.grid': True})
plt.rc('legend', frameon=False)
#params = {'legend.fontsize': 25,
#			'figure.figsize': (12, 9),
#			'axes.labelsize': 25,
#			'axes.titlesize': 25,
#			'xtick.labelsize': 'x-large',
#			'ytick.labelsize': 'x-large',
#      'font.size': 14,
#      'axes.grid': True,
#      'legend.frameon': False,}

#pylab.rcParams.update(params)



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
  points = 40
  x = np.arange(0, 1, 1/points)
  y = np.arange(0, 1, 1/points) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)*0.1 # dampened noise
  mock_data = (analytic + noise).ravel()
  degrees = np.arange(1, 6, dtype=int)
  # plot of analytic franke
  plt.style.use('default')
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  surface = ax.plot_surface(x_mesh, y_mesh, analytic, cmap='viridis')
  fig.colorbar(surface, shrink=0.6)#aspect=20)#, shrink=0.5, aspect=5)
  ax.view_init(elev=15, azim=-7)
  ax.set_zlim(-0.10, 1.40)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.tight_layout()
  # plt.show()
  fig.savefig("figs/FrankeFunction/franke_function_wo_noise.pdf", bbox_inches='tight')
  # plot of franke with noise
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  surface = ax.plot_surface(x_mesh, y_mesh, mock_data.reshape(x_mesh.shape),
                  cmap='viridis')
  fig.colorbar(surface, shrink=0.6)
  # ax.tick_params(axis='both', which='major', labelsize=20)
  ax.view_init(elev=15, azim=-7)
  ax.set_zlim(-0.10, 1.40)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.tight_layout()
  # plt.show()
  fig.savefig("figs/FrankeFunction/franke_function_w_noise.pdf", bbox_inches='tight')
  # # plot of 5th degree estimation of franke
  instance = LinearRegression2D(x, y, mock_data)
  features = instance.features_polynomial_xy(5)
  features_train, features_test, seen, unseen = train_test_split(features, mock_data, test_size=0.2)
  beta = instance.ols(features_train, features_test, seen, return_parameters=True)[0]
  predicted = features @ beta
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.plot_surface(x_mesh, y_mesh, predicted.reshape(x_mesh.shape),
                  cmap='viridis')
  fig.colorbar(surface, shrink=0.6, ax=ax)
  ax.view_init(elev=15, azim=-7)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.tight_layout()#rect=(0.2, 0, 1, 1))
  # plt.show()
  fig.savefig("figs/FrankeFunction/franke_function_deg_5_predicted.pdf", bbox_inches='tight')
  #  plot of betas of model estimation for degree 1-5
  plt.style.use('ggplot')
  pylab.rcParams.update(params)
  betas = []
  vars = []
  fig = plt.figure()
  ax = plt.axes()
  color = plt.cm.viridis(np.linspace(0.9, 0,11))
  ax.set_prop_cycle(plt.cycler('color', color))
  for degree in degrees:
    features = instance.features_polynomial_xy(degree)
    features_train, features_test, seen, unseen = train_test_split(features, mock_data, test_size=0.2)
    parameters, predicted = instance.ols(features_train, features_test, seen,
                             return_parameters=True)
    betas.append(parameters)
    # vars.append(variance(predicted))
    vars.append(np.mean( np.var(predicted) ))
  for i, beta in enumerate(betas):
    beta_indexes = np.arange(1, len(beta)+1)
    plt.errorbar(beta_indexes, beta, yerr=np.sqrt(vars[i]), marker='o', linestyle='--', capsize=4, label='d = %d' % (1 + i)) #  alpha=(1/(1 + i )*10)
  ax.set_xticks([i for i in range(1, len(betas[-1])+1)])
  plt.xlabel(r'$\beta$ coefficient number')
  plt.ylabel(r'$\beta$ coefficient value')
  plt.legend(ncol=3, loc='lower right', fontsize='x-large', columnspacing=0.2, frameon=True, framealpha=0.2, shadow=True)
  plt.tight_layout()
  # plt.show()
  plt.savefig('figs/FrankeFunction/franke_betas.pdf')





def franke_simple_mse_and_r2_analysis():
  """
  Compute predicted for franke function mesh with synthetic noise, with OLS,
  Ridge and Lasso regression for complexities spanning one to five degrees,
  and hyperparameter logspace of 10**-4 to 10**4. Evaluate predicted with
  MSE and r2, and visualize model evaluation.


  """
  np.random.seed(2023)
  plt.style.use('ggplot')
  pylab.rcParams.update(params)
  points = 40
  x = np.arange(0, 1, 1/points)
  y = np.arange(0, 1, 1/points) 
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  mock_data = (analytic + 0.1*noise).ravel()
  degrees = np.arange(1, 6, dtype=int)
  hyperparameters = np.logspace(-4,0,5, dtype=float)
  instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters,
                                       center=True, normalize=False)
  regression_methods = [instance.ols, instance.ridge, instance.lasso]
  eval_funcs = [mean_squared_error, r2_score]
  for regression_method in regression_methods:
    for eval_func in eval_funcs:
      eval_model_mesh = \
          instance.evaluate_model_mesh(regression_method, eval_func)
      filename = f"figs/simple_franke_{regression_method.__name__}"
      filename += f"_{eval_func.__name__}.pdf"
      ylabel = convert_to_label(eval_func.__name__)
      if regression_method == instance.ols:
        fig, ax = instance.visualize_ols(eval_model_mesh, ylabel)
      else:
        fig, ax = instance.visualize_mse_ridge(eval_model_mesh, ylabel)
      fig.savefig(filename)


def bootstrap_analysis():
  """
  Compute predicteds for franke function mesh with synthetic noise, with OLS
  for complexities spanning one to five degrees, and hyperparameter logspace of
  10**-4 to 10**4. Evaluate predicteds with MSE and r2, with bootstrapping, and
  visualize model evaluations.


  """
  np.random.seed(2023)
  #x = np.arange(0, 1, 0.05)
  #y = np.arange(0, 1, 0.05) 
  x = np.linspace(0, 1, 20)
  y = np.linspace(0, 1, 20)
  x_mesh, y_mesh = np.meshgrid(x, y)
  analytic = franke_function(x_mesh, y_mesh)
  noise = np.random.normal(0, 1, x_mesh.shape)
  mock_data = (analytic + 0.1*noise).ravel()
  degrees = np.arange(1, 12, dtype=int)
  hyperparameters = np.logspace(-4,4,10, dtype=float)
  n_bootstraps = 100
  instance = LinearRegression2D(x, y, mock_data,
                                       degrees, hyperparameters)
  instance.ols(features_train= self.features_train, features_test=self.features_train)                                       
  regression_methods = [instance.ols, instance.ridge]
  
  eval_funcs = [mean_squared_error_bootstrapped, bias, variance] # unødvendig
  eval_func_names = ['MSE', 'bias', 'variance']
  eval_model_mesh = \
        instance.evaluate_model_mesh_bootstrap(instance.ols, eval_funcs, # eval funcs unødvendig argument, funka ikke å iterere over funksjoner
                                               n_bootstraps)
  filename = f"figs/simple_franke_ols_"
  fig, ax = plt.subplots(figsize=my_figsize())
  for i, eval_func in enumerate(eval_func_names):
    ax.plot(degrees, eval_model_mesh[i,:], label=eval_func)
    filename += f"{eval_func}_"
  ax.legend()
  ax.set_xlabel("Polynomial degree")
  ax.set_ylabel("Error")
  fig.tight_layout()
  filename += f"{n_bootstraps}_bootstraps.pdf"
  plt.show()
  fig.savefig(filename)
  
  #for i, eval_func in enumerate(eval_funcs):
    #filename = f"figs/simple_franke_ols_{eval_func.__name__}_100_bootstraps.pdf"
    #ylabel = convert_to_label(eval_func.__name__)
    #fig, ax = instance.visualize_ols(eval_model_mesh[i,:], ylabel)
    #plt.show()
    #fig.savefig(filename)


def cross_validation_analysis():
  """
  Compute predicted for franke function mesh with synthetic noise, With ols,
  Ridge, and Lasso regression, for complexities spanning one to five degrees,
  and logspaced hyperparameter spanning 10**-4 to 10**4. Evaluate predicteds
  with MSE, with bootstrapping, and visualize mean of MSEs
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
  mean_mses = np.empty_like(k_folds, dtype=float) # unødvendig?
  regression_methods = [instance.ols, instance.ridge, instance.lasso]
  for regression_method in regression_methods:
    fig, ax = plt.subplots(figsize=my_figsize())
    for i, k_fold in enumerate(k_folds):
      mses = instance.evaluate_model_mesh_cross_validation(regression_method,
                                                           mean_squared_error,
                                                           k_fold)
      ax.plot(degrees, mses, label=f"{k_fold}")
    label = convert_to_label(regression_method.__name__)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Mean MSE")
    fig.legend()
    fig.tight_layout()
    fig.savefig(f"figs/crossval_analysis_mse_{label}.pdf")
    plt.clf()
    plt.close(fig)





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
  hyperparameters = np.logspace(-4,4,10, dtype=float)
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
  # warnings.filterwarnings('ignore', category=ConvergenceWarning)
  #simple_degree_analysis()
  franke_simple_mse_and_r2_analysis()
  #bootstrap_analysis()
  #cross_validation_analysis()
  #franke()
  #terrain()
  #total_mses_franke()
  #total_mses_terrain()