"""
This model contains the regression functions pertaining to project 1.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utilities import my_figsize
from typing import Tuple


def mean_squared_error(y: np.ndarray, model: np.ndarray) -> float:
  """
  Compute mean squared error for some model w.r.t  its analytical expression.


  Parameters
  ----------
  y: x-dimensional array of floats
    Analytical expression
  model: x-dimensional array of floats
    Model
      

  Returns
  -------
      Mean squared error for model.
  """
  err_msg = f"{y.shape=} and {model.shape=}, expected same shapes."
  assert y.shape == model.shape, err_msg
  return np.mean((y - model)**2)


def r2_score(y: np.ndarray, model: np.ndarray) -> float:
  """
  Compute R2-score for some model w.r.t its analytical expression.


  Parameters
  ----------
  y: x-dimensional array of floats
    Analytical expression.
  model: x-dimensional array of floats
    Model.
      

  Returns
  -------
    R2-score for the model.
  """
  err_msg = f"{y.shape=} and {model.shape=}, expected same shapes."
  assert y.shape == model.shape, err_msg
  mse = mean_squared_error(y, model)
  mean_y = np.mean(y)*np.ones_like(y)
  return 1 - mse**2/mean_squared_error(y, mean_y)


class LinearRegression2D:
  """
  TBA
  """
  def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray=None,
               degrees: np.ndarray=None, hyperparameters: np.ndarray=None,
               center=True):
    """
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

  # def features_polynomial_xy(self, degree: int) -> np.ndarray:
  #   """
  #   Construct design matrix for two-dimensional polynomial, where columns are:
  #   (1 + y + ... y**p) + x(1 + y + ... y**p) + ... + x**p(1 + y + ... y**p),
  #   where p is degree for polynomial, x = x_i and y = y_j, indexed such that
  #   row index k  =  len(y)*i + j.


  #   Parameters
  #   ----------
  #   degree: int
  #     Polynomial degree for model.


  #   Returns
  #   -------
  #   two-dimensional-array
  #     Design matrix for two dimensional polynomial of specified degree. 

    
  #   """
  #   assert isinstance(degree, (int, np.int64))
  #   features_xy = np.empty((len_x*len_y, (degree+1)**2), dtype=float)
  #   for i, x_ in enumerate(x):
  #     for j, y_ in enumerate(y):
  #       row = len_y*i + j
  #       for k in range(degree + 1):
  #         for l in range(degree + 1):
  #           col = k*(degree+1) + l
  #           features_xy[row, col] = x_**k*y_**l
  #   if self.center:
  #     features_xy -= np.mean(features_xy, axis=1, keepdims=True)
  #   return features_xy

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
    features_xy = np.empty((len_x*len_y, int((degree+1)*(degree+2)/2)), dtype=float)
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

  def ols(self, degree: int) -> Tuple[np.ndarray, float, float]:
    """
    Partition data into training set and test set, compute optimal parameters 
    with ordinary least squares regression using training set, and compute
    mean squared error and R-squared score for trained model with test set. 


    Parameters
    ----------
    degree: float
      degree of complexity for polynomial.
    
    
    Returns
    -------
    Tuple[np.ndarray, float, float]
        predicted_model: numpy.ndarray
            Predicted values for the test set.
        mse: float
            Mean Squared Error.
        r2: float
            R-squared score.
   
    
    """
    assert self.z is not None, "specifying z is required for  ols"
    features = self.features_polynomial_xy(degree)
    features_train, features_test, z_train, z_test = train_test_split(
        features, self.z_flat)
    optimal_parameters = np.linalg.pinv(
        np.transpose(features_train) @ features_train
    ) @ np.transpose(features_train) @ z_train
    predicted_model = features_test @ optimal_parameters
    mse = mean_squared_error(z_test, predicted_model)
    r2 = r2_score(z_test, predicted_model)
    return predicted_model, mse, r2

  def ridge(self, degree: int, hyperparameter: float
            ) -> Tuple[np.ndarray, float, float]:
    """
    Partition data into training set and test set, compute optimal parameters 
    with ridge regression using training set, and compute mean squared error
    and R-squared score for trained model with test set. 

    
    Parameters
    ----------
    degree: int
      degree of complexity for polynomial.
    
    hyperparameter: float
      hyperparameter
      

    Returns
    -------
    numpy.ndarray
    Tuple[np.ndarray, float, float]
        predicted_model: numpy.ndarray
            Predicted values for the test set.
        mse: float
            Mean Squared Error.
        r2: float
            R-squared score.


    """
    assert self.z is not None, "specifying z is required for  ridge"
    features = self.features_polynomial_xy(degree)
    features_train, features_test, z_train, z_test = train_test_split(
        features, self.z_flat)
    optimal_parameters = np.linalg.pinv(
        np.transpose(features_train) @ features_train
        + np.identity(features_train.shape[1])*hyperparameter
    ) @ np.transpose(features_train) @ z_train
    predicted_model = features_test @ optimal_parameters
    mse = mean_squared_error(z_test, predicted_model)
    r2 = r2_score(z_test, predicted_model)
    return predicted_model, mse, r2

  def lasso(self, degree: int, hyperparameter: float
            ) -> Tuple[np.ndarray, float, float]:
    """
    Partition data into training set and test set, compute optimal parameters 
    with ridge regression using training set, and compute mean squared error
    and R-squared score for trained model with test set. 

    
    Parameters
    ----------
    degree: int
      degree of complexity for polynomial.
    
    hyperparameter: float
      hyperparameter
      

    Returns
    -------
    numpy.ndarray
    Tuple[np.ndarray, float, float]
        predicted_model: numpy.ndarray
            Predicted values for the test set.
        mse: float
            Mean Squared Error.
        r2: float
            R-squared score.


    """
    assert self.z is not None, "specifying z is required for lasso"
    features = self.features_polynomial_xy(degree)
    features_train, features_test, z_train, z_test = train_test_split(
        features, self.z_flat)
    model = Lasso()
    model.fit(features_train, z_train)
    predicted_model = model.predict(features_test)
    mse = mean_squared_error(z_test, predicted_model)
    r2 = r2_score(z_test, predicted_model)
    return predicted_model, mse, r2 

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
      model, mse, r2 = self.ols(degree)
      self.mses_ols[i] = mse
      self.r2s_ols[i] = r2

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
      for j, hyperparameter in enumerate(self.hyperparameters):
        model, mse, r2 = self.ridge(degree, hyperparameter)
        self.mses_ridge[i, j] = mse
        self.r2s_ridge[i, j] = r2
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
      for j, hyperparameter in enumerate(self.hyperparameters):
        model, mse, r2 = self.lasso(degree, hyperparameter)
        self.mses_lasso[i, j] = mse
        self.r2s_lasso[i, j] = r2
    return self.mses_lasso, self.r2s_lasso

  def visualize_mse_ols(self, show=False, save=True):
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

def test_polynomial_features_xy():
  # simplest case test
  x = np.array([1],dtype=float)
  y = np.array([2],dtype=float)
  simple_instance = LinearRegression2D(x, y,center=False)
  degree = 1
  expected = np.array([[1, 2, 1]], dtype=float)
  computed = simple_instance.features_polynomial_xy(degree)
  assert np.allclose(expected,computed)
  assert expected.shape == computed.shape
