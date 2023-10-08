# -*- coding: utf-8 -*-
"""
This model contains the regression functions pertaining to project 1.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utilities import (my_figsize)
from model_evaluation_metrics import (mean_squared_error, r2_score, bias,
                                      variance)
from typing import Tuple


class LinearRegression2D:
  """
  Toolbox for creating predictions with OLS, Ridge and Lasso regression,
  and evaluating and visualizing their MSE and R2
  """

  def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray = None,
               degrees: np.ndarray = None, hyperparameters: np.ndarray = None,
               center=True):
    """
    Instantiate LinearRegression2D object.

    Parameters
    ----------
    x : array like, shape (n), dtype float
      X-dimension mesh
    y : array like, shape (m), dtype float
      Y-dimension mesh.
    z : array like, shape (n, m), dtype float
      Mesh function z(x,y)
    degrees : one-dimensional array of integers
      complexities span for linear regression models.
    hyperparameters : one-dimensional array of floats
      hyperparameter span for linear regression models.
    center: Bool
      True if centering features.

    
    """
    # assert len(x.shape) == 1, "requires m dimensional array."
    # assert len(y.shape) == 1, "requires n dimensional array."
    if degrees is not None:
      err_msg = f"{degrees.dtype=}, must be integer."
      assert np.issubdtype(degrees.dtype, np.integer), err_msg
    if z is not None:
      err_msg = f"{z.shape=}, requires mxn dimensional array."
      assert z.shape == (x.shape[0], y.shape[0]), err_msg
      self.z_flat = z.ravel()
    self.x = x
    self.y = y
    self.z = z
    self.degrees = degrees
    self.hyperparameters = hyperparameters
    self.center = center
    self.initialize_features_n_train_test_split_data()

  def features_polynomial_xy(self, degree: int) -> np.ndarray:
    """
    Construct design matrix for two-dimensional polynomial, where columns are:
    (1 + y + ... y**p) + x(1 + y + ... y**p-1) + ... + x**p,
    where p is degree for polynomial, x = x_i and y = y_j, indexed such that
    row index k  =  len(y)*i + j.


    Parameters
    ----------
    degree : int
      Polynomial degree for model.


    Returns
    -------
    np.ndarray, shape (m*n, (degree+1)*(degree+2)/2), dtype float
      Design matrix for two dimensional polynomial of specified degree. 

    
    """
    assert isinstance(degree, (int, np.int64))
    len_x = self.x.shape[0]
    len_y = self.y.shape[0]
    features_xy = np.empty(
        (len_x*len_y, int((degree+1)*(degree+2)/2)), dtype=float)
    for i, x_ in enumerate(self.x):
      for j, y_ in enumerate(self.y):
        row = len_y*i + j
        col_count = 0
        for k in range(degree + 1):
          for l in range(degree + 1 - k):
            features_xy[row, col_count] = x_**k*y_**l
            col_count += 1
    if self.center:
      features_xy -= np.mean(features_xy, axis=1, keepdims=True)
    return features_xy

  def initialize_features_n_train_test_split_data(self, test_size=0.2):
    """
    Initializes feature arrays for specified polynomial degrees, and partitions
    all feature arrays into corresponding training sets and test sets.
    """
    self.features = {}
    self.features_train = {}
    self.features_test = {}
    self.z_train = {}
    self.z_test = {}
    for degree in self.degrees:
      key = str(degree)
      self.features[key] = self.features_polynomial_xy(degree)
      self.features_train[key], self.features_test[key], self.z_train[key], \
          self.z_test[key] = train_test_split(self.features[key], self.z_flat,
                                              test_size=test_size)

  def ols(self, degree: int, initialized_features=True,
          features_train: np.ndarray = None, features_test: np.ndarray = None,
          z_train: np.ndarray = None) -> np.ndarray:
    """
    Implement Ordinary least squares regression for initialized or specified
    training set and test set.

    
    Parameters
    ----------
    degree: float
      Degree of complexity for polynomial.
    initialized_features: Bool
      True if implementing previously computed features
    features_train: two-dimensional array of floats
      Features from training set.
    features_test: two-dimensional array of floats
      Features from test set
    z_train: one-dimensional array of floats
      flattened mesh function from test set
    
    
    Returns
    -------
    np.ndarray
    

    """ 
    if initialized_features:
      assert self.z is not None, """
      initializing z is required for ols with initialized features"""
      features_train = self.features_train[str(degree)]
      features_test = self.features_test[str(degree)]
      z_train = self.z_train[str(degree)]
    else:
      assert z_train is not None, "z_train cannot be None"
      assert features_test is not None, "features_test cannot be None"
      assert features_train is not None, "features_train cannot be None"
    optimal_parameters = np.linalg.pinv(
        np.transpose(features_train) @ features_train
    ) @ np.transpose(features_train) @ z_train
    predicted = features_test @ optimal_parameters
    return predicted

  def ridge(self, degree: int, hyperparameter: float, 
            initialized_features=True, features_train: np.ndarray = None, 
            features_test: np.ndarray = None, z_train: np.ndarray = None
            ) -> Tuple[np.ndarray, float, float]:
    """
    Implement Ridge regression for initialized or specified training set and
    test set.

    
    Parameters
    ----------
    degree: int
      Degree of complexity for polynomial.
    
    hyperparameter: float
      Hyperparameter.
    
    initialized_features: Bool
      True if implementing previously computed features.
    
    features_train: array_like, shape (n_samples, n_features)
      Features from training set.
    
    features_test: array_like, shape (n_samples, n_features)
      Features from test set.
    
    z_train: array_like, shape (n_samples)
      flattened mesh function from test set.

    
    Returns
    -------
    numpy.ndarray, shape (n_samples), dtype float


    """
    if initialized_features:
      assert self.z is not None, """
      initializing z is required for ols with initialized features"""
      features_train = self.features_train[str(degree)]
      features_test = self.features_test[str(degree)]
      z_train = self.z_train[str(degree)]
    else:
      assert z_train is not None, "z_train cannot be None"
      assert features_test is not None, "features_test cannot be None"
      assert features_train is not None, "features_train cannot be None"
    optimal_parameters = np.linalg.pinv(
        np.transpose(features_train) @ features_train
        + np.identity(features_train.shape[1])*hyperparameter
    ) @ np.transpose(features_train) @ z_train
    predicted = features_test @ optimal_parameters
    return predicted

  def lasso(self, degree: int, hyperparameter: float,
            initialized_features=True, features_train: np.ndarray = None,
            features_test: np.ndarray = None, z_train: np.ndarray = None
            ) -> np.ndarray:
    """
    Implement lasso regression for initialized or specified training set and
    test set.

    
    Parameters
    ----------
    degree: int
      Degree of complexity for polynomial.
    
    hyperparameter: float
      Hyperparameter.
    
    initialized_features: Bool
      True if implementing previously computed features.
    
    features_train: array_like, shape (n_samples, n_features)
      Features from training set.
    
    features_test: array_like, shape (n_samples, n_features)
      Features from test set.
    
    z_train: array_like, shape (n_samples)
      flattened mesh function from test set.

    
    Returns
    -------
    numpy.ndarray, shape (n_samples), dtype float


    """
    if initialized_features:
      assert self.z is not None, """
      initializing z is required for ols with initialized features"""
      features_train = self.features_train[str(degree)]
      features_test = self.features_test[str(degree)]
      z_train = self.z_train[str(degree)]
    else:
      assert z_train is not None, "z_train cannot be None"
      assert features_test is not None, "features_test cannot be None"
      assert features_train is not None, "features_train cannot be None"
    model = Lasso(alpha=hyperparameter)
    model.fit(features_train, z_train)
    predicted = model.predict(features_test)
    return predicted

  def evaluate_predicted_ols(self, model_eval_func: callable) -> dict:
    """
    Compute any of the following model evaluations:
    - MSE
    - R2
    - Bias
    - Variance
    for OLS predictions as a function of degree.

    Parameters
    ----------
    model_eval_func : {mean_squared_error, r2_score, bias, variance}
      Model evaluation function.

    
    Returns
    -------
    np.ndarray
      Model evaluation for OLS predictions as a function of degree.


    """

    model_eval_funcs = [mean_squared_error, r2_score, bias, variance]
    err_msg = "model_eval_func not a permitted Model evaluation callable"
    assert model_eval_func in model_eval_funcs, err_msg
    err_msg = "specying degrees is required for using evaluate_predicted_ols"
    assert self.degrees is not None, err_msg
    evaluated_model = np.empty_like(self.degrees, dtype=float)
    for i, degree in enumerate(self.degrees):
      unseen = self.z_test[str(degree)]
      predicted = self.ols(degree)
      try:
        evaluated_model[i] = model_eval_func(unseen, predicted)
      except TypeError:
        evaluated_model[i] = model_eval_func(predicted)
    return evaluated_model


  def evaluate_predicted_ridge(self, model_eval_func: callable):
    """
    Compute any of the following model evaluations:
    - MSE
    - R2
    - Bias
    - Variance
    for Ridge predictions as a function of degree.

    Parameters
    ----------
    model_eval_func : {mean_squared_error, r2_score, bias, variance}
      Model evaluation function.

    
    Returns
    -------
    np.ndarray
      Model evaluation for Ridge predictions as a function of degree.

      
    """
    model_eval_funcs = [mean_squared_error, r2_score, bias, variance]
    err_msg = "model_eval_func not a permitted Model evaluation callable"
    assert model_eval_func in model_eval_funcs, err_msg
    err_msg = """specifying degrees and hyperparameters is required for using 
    mse_and_r2_ridge"""
    assert self.degrees is not None, err_msg
    evaluated_model = np.empty(
        (self.degrees.shape[0], self.hyperparameters.shape[0]), dtype=float)
    for i, degree in enumerate(self.degrees):
      unseen = self.z_test[str(degree)]
      for j, hyperparameter in enumerate(self.hyperparameters):
        predicted = self.ridge(degree, hyperparameter)
        evaluated_model[i, j] = model_eval_func(unseen, predicted)        
    return evaluated_model

  def evaluate_predicted_lasso(self, model_eval_func: callable):
    """
    Compute any of the following model evaluations:
    - MSE
    - R2
    - Bias
    - Variance
    for Lasso predictions as a function of degree.

    Parameters
    ----------
    model_eval_func : {mean_squared_error, r2_score, bias, variance}
      Model evaluation function.

    
    Returns
    -------
    np.ndarray
      Model evaluation for Lasso predictions as a function of degree.

      
    """
    model_eval_funcs = [mean_squared_error, r2_score, bias, variance]
    err_msg = "model_eval_func not a permitted Model evaluation callable"
    assert model_eval_func in model_eval_funcs, err_msg
    err_msg = """specifying degrees and hyperparameters is required for using 
    mse_and_r2_lasso"""
    assert self.degrees is not None, err_msg
    evaluated_model = np.empty(
        (self.degrees.shape[0], self.hyperparameters.shape[0]), dtype=float)
    for i, degree in enumerate(self.degrees):
      unseen = self.z_test[str(degree)]
      for j, hyperparameter in enumerate(self.hyperparameters):
        predicted = self.lasso(degree, hyperparameter)
        evaluated_model[i, j] = model_eval_func(unseen, predicted) 
    return evaluated_model

  def visualize_quantities_ols(self, show=False, save=True):
    """
    Visualize MSE as function of polynomial degree for prediction by OLS
    regression.


    Parameters
    ----------
    show: Bool
      Shows figure if true.
    save: Bool
      saves figure if true.


    """
    model_eval_str = ["mse", "r2", "bias", "variance"]
    model_eval_funcs = [mean_squared_error, r2_score, bias, variance]
    fig, ax = plt.subplots(figsize=my_figsize())
    for i, model_eval_func in enumerate(model_eval_funcs):
      model_eval = self.evaluate_predicted_ols(model_eval_func)
      ax.plot(self.degrees, model_eval, label=model_eval_str[i])

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("")
    ax.legend()
    fig.tight_layout()
    if show == True:
      plt.show()
    if save == True:
      fig.savefig("figs/ols_everything.pdf")

  def visualize_mse_ridge(self, show=False, save=True):
    """
    Visualize MSE as a function of polynomial degree and hyperparameter for
    prediction by Ridge regression.

    
    Parameters
    ----------
    show: Bool
      Shows figure if true.
    save: Bool
      saves figure if true.

      
    """

    if not hasattr(self, 'mses_ridge'):
      self.mse_and_r2_ridge()
    fig, ax = plt.subplots(figsize=my_figsize())
    degrees_mesh, hyperparameters_mesh = np.meshgrid(
        self.degrees, self.hyperparameters)
    levels = np.linspace(self.mses_ridge.min(), self.mses_ridge.max(), 7)
    contour = ax.contourf(
        degrees_mesh, hyperparameters_mesh, self.mses_ridge.T, levels=levels)
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"hyperparameter $\lambda$")
    ax.grid()
    def format_func(x, _): return f"{x:.2f}"
    cbar = plt.colorbar(contour, format=format_func)
    cbar.set_label('MSE')
    fig.tight_layout()
    if show == True:
      plt.show()
    if save == True:
      fig.savefig("figs/ridge_mse.pdf")
    

  def visualize_mse_lasso(self, show=False, save=True):
    """
    Visualize MSE as a function of polynomial degree and hyperparameter for
    prediction by Lasso regression.

    
    Parameters
    ----------
    show: Bool
      Shows figure if true.
    save: Bool
      saves figure if true.

      
    """
    if not hasattr(self, 'mses_lasso'):
      self.mse_and_r2_lasso()
    fig, ax = plt.subplots(figsize=my_figsize())
    degrees_mesh, hyperparameters_mesh = np.meshgrid(
        self.degrees, self.hyperparameters)
    levels = np.linspace(self.mses_lasso.min(), self.mses_lasso.max(), 7)
    contour = ax.contourf(
        degrees_mesh, hyperparameters_mesh, self.mses_lasso.T, levels=levels)
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"hyperparameter $\lambda$")
    ax.grid()
    def format_func(x, _): return f"{x:.2f}"
    cbar = plt.colorbar(contour, format=format_func)
    cbar.set_label('MSE')
    fig.tight_layout()
    if show == True:
      plt.show()
    if save == True:
      fig.savefig("../plots/lasso_mse.pdf")

  def plot_terrain(self):
      """ Plot entire terrain dataset """
      fig, ax = plt.subplots()
      plt.title('Terrain')
      # x_mesh, y_mesh = np.meshgrid(self.x, self.y)
      ax.imshow(self.z, cmap='viridis')
      plt.xlabel('X')
      plt.ylabel('Y')
      plt.show()

  def plot_terrain_3D(self):
      """ Plot 3D terrain of zoomed in area """
      fig = plt.figure()
      ax = plt.axes(projection = '3d')
      plt.title('Terrain 3D')
      x, y = np.meshgrid(self.x, self.y)
      self.z = np.concatenate(self.z, axis=None)
      z_plot = np.array_split(self.z, len(self.x))
      z_plot = np.array(z_plot)
      surf = ax.plot_surface(x, y, z_plot, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
      fig.colorbar(surf, shrink=0.5, aspect=5)
      plt.show()


def test_polynomial_features_xy():
  """
  Verifies that output of polynomial_features_xy is as expected, with
  comparison of expected and computed output given simple parameters.


  """
  # simplest case test
  x = np.array([1], dtype=float)
  y = np.array([2], dtype=float)
  simple_instance = LinearRegression2D(x, y, center=False)
  degree = 1
  expected = np.array([[1, 2, 1]], dtype=float)
  computed = simple_instance.features_polynomial_xy(degree)
  assert np.allclose(expected, computed)
  assert expected.shape == computed.shape
