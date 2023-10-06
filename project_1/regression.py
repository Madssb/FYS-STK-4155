# -*- coding: utf-8 -*-
"""
This model contains the regression functions pertaining to project 1.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utilities import (my_figsize, mean_squared_error, r2_score)
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
    x: one-dimensional array of floats
      X-dimension mesh
    y: one-dimensional array of floats
      Y-dimension mesh.
    z: two-dimensional array of floats
      Mesh function z(x,y)
    degrees: one-dimensional array of integers
      complexities span for linear regression models.
    hyperparameters: one-dimensional array of floats
      hyperparameter span for linear regression models.
    center: Bool
      True if centering features.

    
    """
    assert len(x.shape) == 1, "requires m dimensional array."
    assert len(y.shape) == 1, "requires n dimensional array."
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
    degree: int
      Polynomial degree for model.


    Returns
    -------
    two-dimensional-array
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
    predicted_model = features_test @ optimal_parameters
    return predicted_model

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
    predicted_model = features_test @ optimal_parameters
    return predicted_model

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
    predicted_model = model.predict(features_test)
    return predicted_model

  def mse_and_r2_ols(self):
    """
    Compute mean squared error for ordinary least square regression computed
    models as a function of model complexity.
    """
    err_msg = "specying degrees is required for using mse_and_r2_ols"
    assert self.degrees is not None, err_msg
    self.mses_ols = np.empty_like(self.degrees, dtype=float)
    self.r2s_ols = np.empty_like(self.degrees, dtype=float)
    for i, degree in enumerate(self.degrees):
      unseen = self.z_test[str(degree)]
      prediction = self.ols(degree)
      self.mses_ols[i] = mean_squared_error(unseen, prediction)
      self.r2s_ols[i] = r2_score(unseen, prediction)

  def mse_and_r2_ridge(self):
    """
    Compute mean squared error for Ridge regression computed models as a
    function of model complexity.
    """
    err_msg = """specifying degrees and hyperparameters is required for using 
    mse_and_r2_ridge"""
    assert self.degrees is not None, err_msg
    self.mses_ridge = np.empty(
        (self.degrees.shape[0], self.hyperparameters.shape[0]), dtype=float)
    self.r2s_ridge = np.empty_like(self.mses_ridge, dtype=float)
    for i, degree in enumerate(self.degrees):
      unseen = self.z_test[str(degree)]
      for j, hyperparameter in enumerate(self.hyperparameters):
        prediction = self.ridge(degree, hyperparameter)            
        self.mses_ridge[i, j] = mean_squared_error(unseen, prediction)
        self.r2s_ridge[i, j] = r2_score(unseen, prediction)
    return self.mses_ridge, self.r2s_ridge

  def mse_and_r2_lasso(self):
    """
    Compute mean squared error for Lasso regression computed models as a
    function of model complexity.
    """
    err_msg = """specifying degrees and hyperparameters is required for using 
    mse_and_r2_lasso"""
    assert self.degrees is not None, err_msg
    self.mses_lasso = np.empty(
        (self.degrees.shape[0], self.hyperparameters.shape[0]), dtype=float)
    self.r2s_lasso = np.empty_like(self.mses_lasso, dtype=float)
    for i, degree in enumerate(self.degrees):
      unseen = self.z_test[str(degree)]
      for j, hyperparameter in enumerate(self.hyperparameters):
        prediction = self.lasso(degree, hyperparameter)
        self.mses_lasso[i, j] = mean_squared_error(unseen, prediction)
        self.r2s_lasso[i, j] = r2_score(unseen, prediction)
    return self.mses_lasso, self.r2s_lasso

  def visualize_mse_ols(self, show=False, save=True):
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
    if not hasattr(self, 'mses_ols'):
      self.mse_and_r2_ols()
    fig, ax = plt.subplots(figsize=my_figsize())
    ax.plot(self.degrees, self.mses_ols, label="MSE")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    if show == True:
      plt.show()
    if save == True:
      fig.savefig("../plots/ols_mse.pdf")

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
      fig.savefig("../plots/ridge_mse.pdf")

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

  def cross_validation(self, degree: int, k: int, method='ols', hyperparameter=None):
    features = self.features_polynomial_xy(degree)
    n = len(self.z_flat)
    mses = np.empty(k)
    r2s = np.empty(k)

    # shuffle indices
    idx = np.random.permutation(np.arange(n))
    # split indices into k groups
    idx_groups = np.array_split(idx, k)

    for g in range(k):
      test_idx = idx_groups[g]
      train_idx = np.concatenate([idx_groups[h] for h in range(k) if h != g])

      features_train = features[train_idx,:]
      features_test = features[test_idx,:]
      z_train = self.z_flat[train_idx]
      z_test = self.z_flat[test_idx]

      if method == 'ols':
        prediction = self.ols(degree, initialized_features=False, 
          features_train=features_train, features_test=features_test, z_train=z_train)
      elif method == 'ridge':
        assert hyperparameter is not None, "hyperparameter cannot be None"
        prediction = self.ridge(degree, hyperparameter, initialized_features=False, 
          features_train=features_train, features_test=features_test, z_train=z_train)
      elif method == 'lasso':
        assert hyperparameter is not None, "hyperparameter cannot be None"
        prediction = self.lasso(degree, hyperparameter, initialized_features=False, 
          features_train=features_train, features_test=features_test, z_train=z_train)
      else:
        print("Choose method 'ols', 'ridge' or 'lasso'")
      mses[g] = mean_squared_error(z_test, prediction)
      r2s[g] = r2_score(z_test, prediction)

    mse_cv = np.mean(mses)    
    r2_cv = np.mean(r2s)
    return mse_cv, r2_cv


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
