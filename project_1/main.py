"""
Solve project here
"""
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import warnings
import seaborn as sns 
sns.set_theme()

from model_evaluation_metrics import (mean_squared_error, r2_score, bias,
                                      variance, mean_squared_error_bootstrapped)
from utilities import (franke_function, convert_to_label)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from regression import LinearRegression2D
from PIL import Image

# formatting plots and figures
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'axes.grid': True})
plt.rc('legend', frameon=False)
params = {'legend.fontsize': 25,
			'figure.figsize': (12, 9),
			'axes.labelsize': 25,
			'axes.titlesize': 25,
			'xtick.labelsize': 'x-large',
			'ytick.labelsize': 'x-large',
     'font.size': 14,
     'axes.grid': True,
     'legend.frameon': False,}

#pylab.rcParams.update(params)



def make_figs_for_everything(instance: LinearRegression2D, data: np.ndarray,
                             data_str: str):
  """
  Visualize all. 
  Note: Depricated function as the ones below are used for more task specific plot making.
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



def simple_degree_analysis(terrain=False):
  """
  Function to investigate Franke function w/w.o noise and Terrain data 
  for a polynomial degree range of 1 -5 . Computes fit with ols and visualizes
  the beta values coefficients for the different degree fits.

  Argument:
    terrain : (bool) default=False
      to choose if to analyse Franke function or terrain data
  Returns:
    None
  """

  np.random.seed(2023)
  if terrain == True:
    data = np.array(Image.open('data/SRTM_data_Norway_1.tif'))
    x_pos, y_pos = 500, 500
    reduce_factor = 30
    y_shift = 1000
    x_shift = 1000
    data = data[y_pos:y_pos+y_shift, x_pos:x_pos+x_shift]
    data = data[::reduce_factor, ::reduce_factor]
    x = np.linspace(0, 1, data.shape[0])
    y = np.linspace(0, 1, data.shape[1])
    data = data.ravel().astype(np.float64)
    degrees = np.arange(1, 6)
    instance = LinearRegression2D(x, y, data, degrees)
  else:
    points = 40
    x = np.arange(0, 1, 1/points)
    y = np.arange(0, 1, 1/points) 
    x_mesh, y_mesh = np.meshgrid(x, y)
    analytic = franke_function(x_mesh, y_mesh)
    noise = np.random.normal(0, 1, x_mesh.shape)*0.1 # dampened noise
    data = (analytic + noise).ravel()
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
    surface = ax.plot_surface(x_mesh, y_mesh, data.reshape(x_mesh.shape),
                    cmap='viridis')
    fig.colorbar(surface, shrink=0.6)
    ax.view_init(elev=15, azim=-7)
    ax.set_zlim(-0.10, 1.40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()
    # plt.show()
    fig.savefig("figs/FrankeFunction/franke_function_w_noise.pdf", bbox_inches='tight')
    # # plot of 5th degree estimation of franke
    instance = LinearRegression2D(x, y, data)
    features = instance.features_polynomial_xy(5)
    features_train, features_test, seen, unseen = train_test_split(features, data, test_size=0.2)
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
    plt.tight_layout()
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
    features_train, features_test, seen, unseen = train_test_split(features, data, test_size=0.2)
    parameters, predicted = instance.ols(features_train, features_test, seen,
                             return_parameters=True)
    betas.append(parameters)
    vars.append(np.mean( np.var(predicted) ))
  for i, beta in enumerate(betas):
    beta_indexes = np.arange(1, len(beta)+1)
    plt.errorbar(beta_indexes, beta, yerr=np.sqrt(vars[i]), marker='o', linestyle='--', capsize=4, label='d = %d' % (1 + i)) #  alpha=(1/(1 + i )*10)
  ax.set_xticks([i for i in range(1, len(betas[-1])+1)])
  plt.xlabel(r'$\beta$ coefficient number')
  plt.ylabel(r'$\beta$ coefficient value')
  plt.legend(ncol=3, loc='best', fontsize='x-large', columnspacing=0.2, frameon=True, framealpha=0.2, shadow=True)
  plt.tight_layout()
  # plt.show()
  if terrain == True:
    plt.savefig('figs/Terrain/terrain_betas.pdf')
  else:
    plt.savefig('figs/FrankeFunction/franke_betas.pdf')
    # plt.show()


def simple_mse_and_r2_analysis(terrain=False):
  """
  Computes fits with OLS, Ridge and Lasso regression for different complexities
  and hyperparameters. Evaluates MSE and r2 of fits and visualizes model evaluations.
  Works on either Franke function with noise or terrain data.

  Argument:
    terrain : (bool) default=False
      to choose if to analyse Franke function or terrain data
  Returns:
    None
  """
  np.random.seed(2023)
  if terrain == True:
    data = np.array(Image.open('data/SRTM_data_Norway_1.tif'))
    x_pos, y_pos = 500, 500
    reduce_factor = 30
    y_shift = 1000
    x_shift = 1000
    data = data[y_pos:y_pos+y_shift, x_pos:x_pos+x_shift]
    data = data[::reduce_factor, ::reduce_factor]
    x = np.linspace(0, 1, data.shape[0])
    y = np.linspace(0, 1, data.shape[1])
    data = data.ravel().astype(np.float64)
    degrees = np.arange(1, 21, dtype=int)
    hyperparameters = np.logspace(-8,0,10, dtype=float)
  else:
    points = 40
    x = np.arange(0, 1, 1/points)
    y = np.arange(0, 1, 1/points) 
    x_mesh, y_mesh = np.meshgrid(x, y)
    analytic = franke_function(x_mesh, y_mesh)
    noise = np.random.normal(0, 1, x_mesh.shape)*0.1 # dampened noise
    data = (analytic + noise).ravel()
    degrees = np.arange(1, 6, dtype=int)
    hyperparameters = np.logspace(-4,4,10, dtype=float)
  instance = LinearRegression2D(x, y, data,
                                       degrees, hyperparameters,
                                       center=True, normalize=True)
  regression_methods = [instance.ols, instance.ridge, instance.lasso]
  eval_funcs = [mean_squared_error, r2_score]
  pylab.rcParams.update(params)
  for regression_method in regression_methods:
    if regression_method == instance.ols:
      fig, ax = plt.subplots(nrows=2, sharex=True)
      ax[0].set_xticks(degrees)
      plt.xlabel("Polynomial degree")
      for i, eval_func in enumerate(eval_funcs):
        eval_model_mesh = \
            instance.evaluate_model_mesh(regression_method, eval_func)
        ylabel = convert_to_label(eval_func.__name__)
        ax[i].plot(degrees, eval_model_mesh, color=f'C{i}')
        if terrain == True:
          filename = f"figs/Terrain/MSE_R2/simple_terrain_{regression_method.__name__}_maxdeg_{np.max(degrees)}"
        else:
          filename = f"figs/FrankeFunction/MSE_R2/simple_franke_{regression_method.__name__}"
        filename += f"_{eval_func.__name__}.pdf"
        ax[i].set_ylabel(ylabel)
      plt.tight_layout()
      fig.savefig(filename)
    else:
      for eval_func in eval_funcs:
        eval_model_mesh = \
            instance.evaluate_model_mesh(regression_method, eval_func)
        if terrain == True:
          filename = f"figs/Terrain/MSE_R2/simple_terrain_{regression_method.__name__}maxdeg_{np.max(degrees)}"
        else:
          filename = f"figs/FrankeFunction/MSE_R2/simple_franke_{regression_method.__name__}"
        filename += f"_{eval_func.__name__}.pdf"
        ylabel = convert_to_label(eval_func.__name__)
        fig, ax = instance.visualize_mse_ridge(eval_model_mesh, ylabel)
        ax.set_xticks(degrees)
        fig.savefig(filename)


def bootstrap_analysis(terrain=False):
  """
  Executes BVT analysis and computes fits with OLS, Ridge and Lasso regression
  with bootstrap resampling implemented.

  Argument:
    terrain : (bool) default=False
      to choose if to analyse Franke function or terrain data
  Returns:
    None
  """
  np.random.seed(2023)
  pylab.rcParams.update(params)
  if terrain == True:
    data = np.array(Image.open('data/SRTM_data_Norway_1.tif'))
    x_pos, y_pos = 500, 500
    reduce_factor = 30
    y_shift = 1000
    x_shift = 1000
    data = data[y_pos:y_pos+y_shift, x_pos:x_pos+x_shift]
    data = data[::reduce_factor, ::reduce_factor]
    x = np.linspace(0, 1, data.shape[0])
    y = np.linspace(0, 1, data.shape[1])
    data = data.ravel().astype(np.float64)
  else:
    points = 20 
    x = np.arange(0, 1, 1/points)
    y = np.arange(0, 1, 1/points)
    x_mesh, y_mesh = np.meshgrid(x, y)
    analytic = franke_function(x_mesh, y_mesh)
    noise = np.random.normal(0, 1, x_mesh.shape)*0.1
    data = (analytic + noise).ravel()
  n_bootstraps = 500
  degrees = np.arange(1, 12, dtype=int)
  instance = LinearRegression2D(x, y, data,
                                       degrees)
  
  eval_funcs = [mean_squared_error_bootstrapped, bias, variance] 
  eval_func_names = ['MSE', 'bias', 'variance']
  eval_model_mesh = \
        instance.evaluate_model_mesh_bootstrap(instance.ols, eval_funcs, n_bootstraps)
  if terrain:
    filename = f"figs/Terrain/bootstrap/BVT_terrain_ols_"
  else:
    filename = f"figs/FrankeFunction/bootstrap/BVT_franke_ols_"
  fig, ax = plt.subplots()
  for i, eval_func in enumerate(eval_func_names):
    ax.plot(degrees, eval_model_mesh[i, :], label=eval_func)
    filename += f"{eval_func}_"
  ax.legend()
  ax.set_xlabel("Polynomial degree")
  ax.set_ylabel("Error")
  ax.set_xticks(degrees[::2])
  fig.tight_layout()
  if terrain:
    filename += f"deg_{np.max(degrees)}_{n_bootstraps}_bootstraps.pdf"
  else:
    filename += f"deg_{np.max(degrees)}_{points}points_{n_bootstraps}_bootstraps.pdf"
  fig.savefig(filename)
  # plt.show()
  # Ridge and Lasso with bootstrap
  if not terrain:
    points = 40 
    x = np.arange(0, 1, 1/points)
    y = np.arange(0, 1, 1/points)
    x_mesh, y_mesh = np.meshgrid(x, y)
    analytic = franke_function(x_mesh, y_mesh)
    noise = np.random.normal(0, 1, x_mesh.shape)*0.1
    data = (analytic + noise).ravel()

  degrees = np.arange(1, 6, dtype=int)
  hyperparameters = np.logspace(-4,4,10, dtype=float)
  instance2 = LinearRegression2D(x, y, data,
                                       degrees, hyperparameters)
  regression_methods = [instance2.ridge, instance2.lasso]
  for regression_method in regression_methods:
    eval_model_mesh = instance2.evaluate_model_mesh_bootstrap(regression_method,
                                                           mean_squared_error,
                                                           n_bootstraps)
    clabel = "MSE"                                                       
    fig, ax = instance2.visualize_mse_ridge(eval_model_mesh[0], clabel)
    ax.set_xticks(degrees)
    label = convert_to_label(regression_method.__name__)
    if terrain:
      filename = f"figs/Terrain/bootstrap/bootstrap_analysis_mse_{label}_bootstraps_{n_bootstraps}_terrain.pdf"
    else:
      filename = f"figs/FrankeFunction/bootstrap/bootstrap_analysis_mse_{label}_bootstraps_{n_bootstraps}_40x40.pdf"
    fig.savefig(filename)
  plt.show()


def cross_validation_analysis(terrain=False):
  """
  Computes fits with OLS, Ridge and Lasso regression using K-folds resampling method.
  Then computes and visualizes the MSE for respective regression method.
  For OLS it examines MSE with kfolds in range 5 - 10.


  Argument:
    terrain : (bool) default=False
      to choose if to analyse Franke function or terrain data
  Returns:
    None
  """
  np.random.seed(2023)
  pylab.rcParams.update(params)
  if terrain == True:
    data = np.array(Image.open('data/SRTM_data_Norway_1.tif'))
    x_pos, y_pos = 500, 500
    reduce_factor = 30
    y_shift = 1000
    x_shift = 1000
    data = data[y_pos:y_pos+y_shift, x_pos:x_pos+x_shift]
    data = data[::reduce_factor, ::reduce_factor]
    x = np.linspace(0, 1, data.shape[0])
    y = np.linspace(0, 1, data.shape[1])
    data = data.ravel().astype(np.float64)
    degrees = np.arange(1, 21, dtype=int)
    hyperparameters = np.logspace(-8,0,10, dtype=float)
    kfold = 5 # for heat maps
  else:
    points = 20 
    x = np.arange(0, 1, 1/points)
    y = np.arange(0, 1, 1/points)
    x_mesh, y_mesh = np.meshgrid(x, y)
    analytic = franke_function(x_mesh, y_mesh)
    noise = np.random.normal(0, 1, x_mesh.shape)*0.1
    data = (analytic + noise).ravel()
    degrees = np.arange(1, 12, dtype=int)
    hyperparameters = np.logspace(-4,4,10, dtype=float)
  instance = LinearRegression2D(x, y, data,
                                       degrees, hyperparameters) 
  k_folds = np.arange(5,11, dtype=int)
  fig, ax = plt.subplots()
  for i, k_fold in enumerate(k_folds):
    mses = instance.evaluate_model_mesh_cross_validation(instance.ols,
                                                           mean_squared_error,
                                                           k_fold)
    ax.plot(degrees, mses, label=f"k: {k_fold}")
  ax.set_ylabel("MSE")
  ax.set_xlabel("Polynomial degree")
  plt.legend(fontsize=25)
  fig.tight_layout()
  if terrain:
    fig.savefig(f"figs/Terrain/crossval/crossval_analysis_mse_ols_terrain_maxdeg_{np.max(degrees)}.pdf")
  else:
    fig.savefig(f"figs/FrankeFunction/crossval/crossval_analysis_mse_ols_maxdeg_{np.max(degrees)}.pdf")
  if not terrain:
    points = 40 
    x = np.arange(0, 1, 1/points)
    y = np.arange(0, 1, 1/points)
    x_mesh, y_mesh = np.meshgrid(x, y)
    analytic = franke_function(x_mesh, y_mesh)
    noise = np.random.normal(0, 1, x_mesh.shape)*0.1
    data = (analytic + noise).ravel()
  instance2 = LinearRegression2D(x, y, data,
                                       degrees, hyperparameters)
  regression_methods = [instance2.ridge, instance2.lasso]
  for regression_method in regression_methods:
    eval_model_mesh = instance2.evaluate_model_mesh_cross_validation(regression_method,
                                                           mean_squared_error,
                                                           k_fold)
    clabel = "MSE"                                                       
    fig, ax = instance2.visualize_mse_ridge(eval_model_mesh, clabel)
    ax.set_xticks(degrees)
    label = convert_to_label(regression_method.__name__)
    if terrain:
      filename = f"figs/Terrain/crossval/crossval_analysis_mse_{label}_kfolds_{kfold}_terrain_maxdeg_{np.max(degrees)}.pdf"
    else:
      filename = f"figs/FrankeFunction/crossval/crossval_analysis_mse_{label}_kfolds_{kfold}_{points}x{points}_maxdeg_{np.max(degrees)}.pdf"
    fig.savefig(filename)
  plt.show()





def terrain_best_fit():
  """
  Visualizes the differences between the different regression method fits again the terrain data.

  Retruns:
    None
  """
  np.random.seed(2023)
  data = np.array(Image.open('data/SRTM_data_Norway_1.tif'))
  x_pos, y_pos = 500, 500
  reduce_factor = 30
  y_shift = 1000
  x_shift = 1000
  z = data[y_pos:y_pos+y_shift, x_pos:x_pos+x_shift]
  z = z[::reduce_factor, ::reduce_factor]
  x = np.linspace(0,1,z.shape[0])
  y = np.linspace(0,1,z.shape[1])
  z = z.ravel().astype(np.float64)
  degree = np.arange(1,21)
  hyperparameter = 10e-8
  instance = LinearRegression2D(x, y, z, degree, hyperparameter, center=False, normalize=False)
  features = instance.features_polynomial_xy(degree.max())
  x_mesh, y_mesh = np.meshgrid(x, y)

  predicted_ols = instance.ols(features, features, z)
  predicted_ridge = instance.ridge(features, features, z, hyperparameter=hyperparameter)
  predicted_lasso = instance.lasso(features, features, z, hyperparameter=hyperparameter)
  
  fig, axs = plt.subplots(2, 4, figsize=(17, 7))#, sharey='row')
  color_3D = plt.cm.coolwarm

  for ax in axs[0, :]:
    ax.axis('off')
  for ax in axs[1, :]:
    ax.axis('off')
  
  axs[0, 0].set_title('Terrain')
  axs[0, 0] = fig.add_subplot(2, 4, 1, projection='3d')
  surface = axs[0, 0].plot_surface(y_mesh, x_mesh, z.reshape(x_mesh.shape),
                  cmap=color_3D)
  axs[0, 0].view_init(elev=13, azim=-9)
  axs[0, 0].set_xticks([]); axs[0, 0].set_yticks([]); axs[0, 0].set_zticks([])
  axs[0, 0].dist = 8

  axs[0, 1].set_title('OLS')
  axs[0, 1] = fig.add_subplot(2, 4, 2, projection='3d')
  axs[0, 1].plot_surface(y_mesh, x_mesh, predicted_ols.reshape(x_mesh.shape),
                  cmap=color_3D)
  axs[0, 1].view_init(elev=13, azim=-9)
  axs[0, 1].set_xticks([]); axs[0, 1].set_yticks([]); axs[0, 1].set_zticks([])
  axs[0, 1].dist = 8

  axs[0, 2].set_title('Ridge')
  axs[0, 2] = fig.add_subplot(2, 4, 3, projection='3d')
  axs[0, 2].plot_surface(y_mesh, x_mesh, predicted_ridge.reshape(x_mesh.shape),
                  cmap=color_3D)
  axs[0, 2].view_init(elev=13, azim=-9)
  axs[0, 2].set_xticks([]); axs[0, 2].set_yticks([]); axs[0, 2].set_zticks([])
  axs[0, 2].dist = 8

  axs[0, 3].set_title('Lasso')
  axs[0, 3] = fig.add_subplot(2, 4, 4, projection='3d')
  axs[0, 3].plot_surface(y_mesh, x_mesh, predicted_lasso.reshape(x_mesh.shape),
                  cmap=color_3D)
  axs[0, 3].view_init(elev=13, azim=-9)
  axs[0, 3].set_xticks([]); axs[0, 3].set_yticks([]); axs[0, 3].set_zticks([])
  axs[0, 3].dist = 8

  axs[1, 0].imshow(z.reshape(x_mesh.shape), cmap='coolwarm')
  axs[1, 1].imshow(predicted_ols.reshape(y_mesh.shape), cmap='coolwarm')
  axs[1, 2].imshow(predicted_ridge.reshape(y_mesh.shape), cmap='coolwarm')
  axs[1, 3].imshow(predicted_lasso.reshape(y_mesh.shape), cmap='coolwarm')

  plt.tight_layout()
  cbar = fig.colorbar(surface, ax=axs[:, 3])#, ticks=[np.min(predicted_ols), np.max(predicted_ols)])
  cbar.set_label('Height above sea level [m]')
  plt.savefig('figs/Terrain/best_fit.pdf', bbox_inches='tight')
  # plt.show()


def total_mses_franke():
  """
  Computse mean MSEs for OLS, Ridge and Lasso models fitting Frankes function
  output with synthetic noise, cross validated and not.

  Note: Depricated, not used to produce main results for paper.
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
  Computes mean MSEs for OLS, Ridge and Lasso models fitting terrain data,
  cross validated and not.

  Note: Depricated, not used to produce main results for paper.
  """
  img = Image.open("data/SRTM_data_Norway_1.tif")
  data = np.array(img)
  n_pts = 1000
  ds_factor = int(np.round(np.sqrt((data.shape[0]*data.shape[1])/n_pts)))
  data_downsampled = data[::ds_factor,::ds_factor]
  print(data_downsampled.shape)
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
  warnings.filterwarnings('ignore', category=ConvergenceWarning) # to quell sklearn lasso's convergence warnings 
  """ generate franke plots and figures """
  # simple_degree_analysis()
  # simple_mse_and_r2_analysis()
  bootstrap_analysis()
  # cross_validation_analysis()
  # generate terrain plots and figures
  """ generate terrain plots and figures """
  # simple_degree_analysis(terrain=True)
  # simple_mse_and_r2_analysis(terrain=True)
  # # # bootstrap_analysis(terrain=True) # not needed for paper here mainly as an option
  # cross_validation_analysis(terrain=True)
  """ Not used funcs below might be useful for specific
    results we don't get from main generating functions """
  #franke()
  #terrain_best_fit()
  #total_mses_franke()
  #total_mses_terrain()