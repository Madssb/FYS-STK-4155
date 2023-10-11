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
               test_size = 0.2, n_subsets = 5, center=True, normalize=True):
    """
    Instantiate LinearRegression2D object.

    Parameters
    ----------
    x : array like, shape (n), dtype float
      X-dimension mesh
    y : array like, shape (m), dtype float
      Y-dimension mesh.
    z : array like, shape (n *m), dtype float
      Mesh function z(x,y), but flattened.
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
      assert len(z.shape) == 1, "z must be flattened"
      assert len(x.shape) == 1, "x must be flat"
      assert len(y.shape) == 1, "y must be flat"
      err_msg = f"{z.shape[0]=} is not { x.shape[0] * y.shape[0]=}"
      assert z.shape[0] == x.shape[0] * y.shape[0], err_msg
    self.x = x
    self.y = y
    self.z = z
    if center:
      self.z -= np.mean(self.z)
    if normalize:
      self.z /= np.std(z)
    self.degrees = degrees
    self.hyperparameters = hyperparameters
    self.test_size = test_size
    self.n_subsets = n_subsets
    self.center = center
    self.normalize = normalize
  

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
    assert isinstance(degree, (int, np.integer))
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
      features_xy -= np.mean(features_xy, axis=0, keepdims=True)
    if self.normalize:
      features_xy[:,1:] /= np.std(features_xy[:,1:], axis=0, keepdims=True)
    return features_xy

  def ols(self, features_train: np.ndarray, features_test: np.ndarray,
          z_train: np.ndarray) -> np.ndarray:
    """
    Implement Ordinary least squares regression for initialized or specified
    training set and test set.

    
    Parameters
    ----------
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
    optimal_parameters = np.linalg.pinv(
        np.transpose(features_train) @ features_train
    ) @ np.transpose(features_train) @ z_train
    predicted = features_test @ optimal_parameters
    return predicted

  def ridge(self, features_train: np.ndarray, 
            features_test: np.ndarray, z_train: np.ndarray, 
            hyperparameter: float) -> Tuple[np.ndarray, float, float]:
    """
    Implement Ridge regression for initialized or specified training set and
    test set.

    
    Parameters
    ----------
    hyperparameter: float
      Hyperparameter.
    
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
    assert isinstance(hyperparameter, float), f"{type(hyperparameter)=}, is not float."
    optimal_parameters = np.linalg.pinv(
        np.transpose(features_train) @ features_train
        + np.identity(features_train.shape[1])*hyperparameter
    ) @ np.transpose(features_train) @ z_train
    predicted = features_test @ optimal_parameters
    return predicted

  def lasso(self, features_train: np.ndarray,
            features_test: np.ndarray, z_train: np.ndarray,
            hyperparameter: float) -> np.ndarray:
    """
    Implement lasso regression for initialized or specified training set and
    test set.

    
    Parameters
    ----------
    hyperparameter: float
      Hyperparameter.
    
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
    assert z_train is not None, "z_train cannot be None"
    assert features_test is not None, "features_test cannot be None"
    assert features_train is not None, "features_train cannot be None"
    model = Lasso(alpha=hyperparameter)
    model.fit(features_train, z_train)
    predicted = model.predict(features_test)
    return predicted
  
  def evaluate_predicted(self, degree: int, hyperparameter: float,
                         regression_method: callable,
                         model_eval_func: callable) -> float:
    """
    Compute  model evaluation quantity for specified regression method.

    Parameters
    ----------
    degree : int
      Polynomial degree.
    hyperparameter : float
      Hyperparameter.
    regression_method : {self.mse, self.ridge, self.lasso}
      Method used to compute prediction.
    model_eval_func : {mean_squared_error, r2_score, bias, variance}
      Function applied for evaluate predction w.r.t unseen data.

    
    Returns
    -------
    np.ndarray
      Model evaluation.


    """
    regression_methods = [self.ols, self.ridge, self.lasso]
    err_msg = "regression_method not method in LinearRegression2D."
    assert regression_method in regression_methods, err_msg
    model_eval_funcs = [mean_squared_error, r2_score, bias, variance]
    err_msg = "model_eval_func not a permitted Model evaluation callable"
    assert model_eval_func in model_eval_funcs, err_msg
    features = self.features_polynomial_xy(degree)
    response = self.z
    features_train, features_test, seen, unseen = \
      train_test_split(features, response, test_size=self.test_size)
    try:
      predicted = regression_method(features_train, features_test, seen,
                                    hyperparameter)
    except TypeError:
      predicted = regression_method(features_train, features_test, seen)
    try:
      model_eval = model_eval_func(unseen, predicted)
    except TypeError:
      model_eval = model_eval_func(predicted)
    return model_eval

  def evaluate_predicted_crossval(self, degree: int, 
                                   hyperparameter: float,
                                   regression_method: callable,
                                   model_eval_func: callable) -> float:
    """
    Compute averaged model evaluation for specified regression method
    with bootstrap.


    Parameters
    ----------
    degree : int
      Polynomial degree.
    hyperparameter : float
      Hyperparameter.
    regression_method : {self.mse, self.ridge, self.lasso}
      Method used to compute prediction.
    model_eval_func : {mean_squared_error, r2_score, bias, variance}
      Function applied for evaluate predction w.r.t unseen data.
    
    
    Returns
    -------
      Model evaluation.


    """
    model_eval_funcs = [mean_squared_error, r2_score, bias, variance]
    err_msg = "model_eval_func not a permitted Model evaluation callable"
    assert model_eval_func in model_eval_funcs, err_msg
    regression_methods = [self.ols, self.ridge, self.lasso]
    err_msg = "regression_method not method in LinearRegression2D."
    assert regression_method in regression_methods, err_msg
    features = self.features_polynomial_xy(degree)
    response = self.z
    shuffled_indices = np.random.permutation(np.arange(response.shape[0]))
    indice_subsets = np.array_split(shuffled_indices, self.n_subsets)
    cumulative_model_eval = 0
    for test_set_indices in indice_subsets:
      training_set_indices = np.delete(shuffled_indices, test_set_indices)
      features_train = features[training_set_indices, :]
      features_test = features[test_set_indices, :]
      seen = response[training_set_indices]
      unseen = response[test_set_indices]
      try:
        predicted = regression_method(features_train, features_test, seen, hyperparameter)
      except TypeError:
        predicted = regression_method(features_train, features_test, seen)
      try:
        cumulative_model_eval += model_eval_func(unseen, predicted)
      except TypeError:
        cumulative_model_eval += model_eval_func(predicted)
      avg_model_eval = cumulative_model_eval/self.n_subsets
      return avg_model_eval

  def evaluate_model_mesh(self, regression_method: callable,
                          model_eval_func: callable,
                          eval_predicted_method: callable) -> np.ndarray:
    """
    Compute model evaluation quantity for specified regression method,
    with/without bootstrap, on degrees and hyperparameters mesh.
    

    Parameters
    ----------
    regression_method : {self.mse, self.ridge, self.lasso}
      Method used to compute prediction.
    model_eval_func : {mean_squared_error, r2_score, bias, variance}
      Function applied for evaluate predction w.r.t unseen data.
    eval_predicted_method : {self.evalute_predicted, self.evaluate_predicted_crossval}
      method for computing model evaluation, i.e. apply bootstrap or not.
    
    
    Returns
    -------
      Model evaluation mesh function.


    """
    eval_predicted_methods = [self.evaluate_predicted,
                              self.evaluate_predicted_crossval]
    err_msg = "eval_predicted_method not method in Linearregression2D"
    assert eval_predicted_method in eval_predicted_methods, err_msg
    if regression_method == self.ols:
      model_eval_array = np.empty_like(self.degrees, dtype=float)
      for i, degree in enumerate(self.degrees):
        model_eval_array[i] = eval_predicted_method(degree, None, 
                                                    regression_method, 
                                                    model_eval_func)
      return model_eval_array
    model_eval_array = np.empty((self.degrees.shape[0],
                                 self.hyperparameters.shape[0]), dtype=float)
    for i, degree in enumerate(self.degrees):
      for j, hyperparameter in enumerate(self.hyperparameters):
         model_eval_array[i, j] = eval_predicted_method(degree, hyperparameter,
                                                        regression_method, 
                                                        model_eval_func)
    return model_eval_array

  def visualize_ols(self, quantity: np.ndarray, ylabel: str, label: str = None):
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
    fig, ax = plt.subplots(figsize=my_figsize())
    ax.plot(self.degrees, quantity, label=label)
    ax.set_xlabel("Polynomial degree")
    if ylabel:
      ax.set_ylabel(ylabel)
    if label:
      ax.legend()
    fig.tight_layout()
    return fig, ax

  def visualize_mse_ridge(self, quantity: np.ndarray, clabel: str):
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
    assert len(quantity.shape) == 2, f"{len(quantity.shape)=} is not 2."
    fig, ax = plt.subplots(figsize=my_figsize())
    degrees_mesh, hyperparameters_mesh = np.meshgrid(
        self.degrees, self.hyperparameters)
    try:
      levels = np.linspace(quantity.min(), quantity.max(), 7)
      contour = ax.contourf(
          degrees_mesh, hyperparameters_mesh, quantity.T, levels=levels)
    except ValueError:
      contour = ax.contourf(
          degrees_mesh, hyperparameters_mesh, quantity.T)  
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"hyperparameter $\lambda$")
    ax.grid()
    def format_func(x, _): return f"{x:.2f}"
    cbar = plt.colorbar(contour, format=format_func)
    cbar.set_label(clabel)
    fig.tight_layout()
    return fig, ax
    
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
