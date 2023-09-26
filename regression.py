"""
This model contains the regression functions pertaining to project 1.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utilities import my_figsize


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



def features_polynomial_xy(x: np.ndarray, y: np.ndarray, degree: int,
                                scale=True) -> np.ndarray:
  """
  Construct design matrix for 2 dim polynomial:
  (1 + y + ... y**p) + x(1 + y + ... y**p) + ... + x**p(1 + y + ... y**p),
  where p is degree for polynomial.


  Parameters
  ----------
  x: one-dimensional array of floats
    X-dimension mesh
  y: one-dimensional array of floats
    Y-dimension mesh.
  polynomial_degree
    Polynomial degree for model.
  scale
    True if scaling data, false if not.


  Returns
  two-dimensional-array
    Design matrix for two dimensional polynomial of specified degree. 

  
  """
  assert len(x.shape) == 1, "requires n dimensional array."
  assert len(y.shape) == 1, "requires n dimensional array."
  assert isinstance(degree, (int, np.int64))
  len_x = x.shape[0]
  len_y = y.shape[0]
  features_xy = np.empty((len_x*len_y, (degree+1)**2), dtype=float)
  for i, x_ in enumerate(x):
    for j, y_ in enumerate(y):
      row = len_y*i + j
      for k in range(degree + 1):
        for l in range(degree + 1):
          col = k*(degree+1) + l
          features_xy[row, col] = x_**k*y_**l
  if scale:
    features_xy -= np.mean(features_xy, axis=1, keepdims=True)
  return features_xy


def test_features_polynomial_xy():
  """ 
  Ensures features_polynomial_xy() is working as intended.
  """
  x = np.array([2, 3])
  y = np.array([4, 5])
  expected_features = np.array([[1, 4, 2, 8],
                                     [1, 5, 2, 10],
                                     [1, 4, 3, 12],
                                     [1, 5, 3, 15]])
  features = features_polynomial_xy(x, y, 1, scale=False)
  assert features.shape == (4, 4)
  print(features)
  assert (features == expected_features).all()


def ols_regression(features: np.ndarray, y: np.ndarray) -> np.ndarray:
  """
  Compute the optimal parameters per Ordinary Least Squares regression.


  Parameters
  ----------
  features: two-dimensional array of floats
    design matrix for n-dimensional mesh
    Two dimensional numpy array
  y: one-dimensional array of floats
    n-dimensional mesh function linearized
  
  
  Returns
  -------
  numpy.ndarray
      Optimal parameters as predicted by Ridge.


  Raises
  ------
  AssertionError
      shapes of features or y are not permitted.

    
  """
  assert len(features.shape) == 2, "requires nxm dimensional array."
  assert len(y.shape) == 1, "requires n dimensional array."
  return np.linalg.pinv(
      np.transpose(features) @ features
  ) @ np.transpose(features) @ y


def ridge_regression(features: np.ndarray, y: np.ndarray,
                     hyperparameter: float) -> np.ndarray:
  """
  Computes the optimal parameters per Ridge regression.

  Parameters
  ----------
  features
    Two-dimensional numpy array
  y
    One-dimensional numpy array
  hyperparameter
    TBA

  Returns
  -------
  numpy.ndarray
    Optimal parameters

  Raises
  ------
  AssertionError
    shapes of features or y are not permitted, or hyperparameter is not
    float.


  """
  assert len(features.shape) == 2, "requires nxm dimensional array."
  assert len(y.shape) == 1, "requires n dimensional array"
  assert isinstance(hyperparameter, float), "must be float"
  return np.linalg.pinv(
      np.transpose(features) @ features
      + np.identity(features.shape[1])*hyperparameter
  ) @ np.transpose(features) @ y




class LinearRegression2D:
  def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
               degrees: np.ndarray, hyperparameters: np.ndarray,
               scale=True):
    """
    Parameters
    ----------
    x: one-dimensional array of floats
      X-dimension mesh
    y: one-dimensional array of floats
      Y-dimension mesh.
    z: two-dimensional array of floats
      Mesh function z(x,y)
    """
    assert len(x.shape) == 1, "requires m dimensional array."
    assert len(y.shape) == 1, "requires n dimensional array."
    err_msg = f"{degrees.dtype=}, must be integer."
    assert np.issubdtype(degrees.dtype, np.integer), err_msg
    err_msg = f"{z.shape=}, requires mxn dimensional array."
    assert z.shape == (x.shape[0], y.shape[0]), err_msg
    self.degrees = degrees
    self.hyperparameters = hyperparameters
    self.x = x
    self.y = y
    self.z = z
    self.z_flat = z.ravel()
    self.scale= scale

  def features_polynomial_xy(
      self, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """
    Construct design matrix for 2 dim polynomial:
    (1 + y + ... y**p) + x(1 + y + ... y**p) + ... + x**p(1 + y + ... y**p),
    where p is degree for polynomial.


    Parameters
    ----------
    x: one-dimensional array of floats
      X-dimension mesh
    y: one-dimensional array of floats
      Y-dimension mesh.
    polynomial_degree
      Polynomial degree for model.
    scale
      True if scaling data, false if not.


    Returns
    two-dimensional-array
      Design matrix for two dimensional polynomial of specified degree. 

    
    """
    assert len(x.shape) == 1, "requires n dimensional array."
    assert len(y.shape) == 1, "requires n dimensional array."
    assert isinstance(degree, (int, np.int64))
    len_x = x.shape[0]
    len_y = y.shape[0]
    features_xy = np.empty((len_x*len_y, (degree+1)**2), dtype=float)
    for i, x_ in enumerate(x):
      for j, y_ in enumerate(y):
        row = len_y*i + j
        for k in range(degree + 1):
          for l in range(degree + 1):
            col = k*(degree+1) + l
            features_xy[row, col] = x_**k*y_**l
    if self.scale:
      features_xy -= np.mean(features_xy, axis=1, keepdims=True)
    return features_xy

  def ols_regression(self, degree: int) -> np.ndarray:
    """
    Compute the optimal parameters per Ordinary Least Squares regression.


    Parameters
    ----------
    degree: float
      degree of complexity for polynomial.
    
    
    Returns
    -------
    numpy.ndarray
        Optimal parameters as predicted by Ridge.

          
    """
    features = self.features_polynomial_xy(self.x, self.y, degree)
    features_train, features_test, z_train, z_test = train_test_split(
      features, self.z_flat)
    optimal_parameters = np.linalg.pinv(
        np.transpose(features_train) @ features_train
    ) @ np.transpose(features_train) @ z_train
    predicted_model = features_test @ optimal_parameters
    mse = mean_squared_error(z_test, predicted_model)
    r2 = r2_score(z_test, predicted_model)
    return predicted_model, mse, r2

  def ridge_regression(self, degree: int, hyperparameter: float) -> np.ndarray:
    """
    Computes the optimal parameters per Ridge regression.

    Parameters
    ----------
    degree: float
      degree of complexity for polynomial.
    
    hyperparameter
      TBA

    Returns
    -------
    numpy.ndarray
      Optimal parameters


    """
    features = self.features_polynomial_xy(self.x, self.y, degree)
    features_train, features_test, z_train, z_test = train_test_split(
      features, self.z_flat)
    optimal_parameters =  np.linalg.pinv(
        np.transpose(features_train) @ features_train
        + np.identity(features_train.shape[1])*hyperparameter
    ) @ np.transpose(features_train) @ z_train
    predicted_model = features_test @ optimal_parameters
    mse = mean_squared_error(z_test, predicted_model)
    r2 = r2_score(z_test, predicted_model)
    return predicted_model, mse, r2
  
  def mse_and_r2_ols(self, visualize=True):
    """
    Compute mean squared error for ordinary least square regression computed
    models as a function of model complexity.
    """
    self.mses_ols = np.empty_like(self.degrees,dtype=float)
    self.r2s_ols = np.empty_like(self.degrees,dtype=float)
    for i, degree in enumerate(self.degrees):
      model, mse, r2 = self.ols_regression(degree)
      self.mses_ols[i] = mse
      self.r2s_ols[i] = r2
    if not visualize:
      return
    

  def visualize_mse(self):
    fig, ax = plt.subplots(figsize=my_figsize())
    ax.plot(self.degrees, self.mses_ols, label="MSE")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("MSE")
    fig.savefig("mse_ols.pdf")  

  def mse_and_r2_ridge(self, visualize=True):
    """
    Compute mean squared error for Ridge regression computed models as a
    function of model complexity.
    """
    self.mses_ridge = np.empty(
      self.degrees.shape[0], self.hyperparameters.shape[0], dtype=float)
    for i, degree in enumerate(self.degrees):
      for j, hyperparameter in enumerate(self.hyperparameters):
        model, mse, r2 = self.ridge_regression(degree, hyperparameter)
        self.mses_ridge[i][j] = mse
    #visualize results
    if not visualize:
      return
    fig, ax = plt.subplots(figsize=my_figsize())
    degrees_mesh, hyperparameters_mesh = np.meshgrid(
      self.degrees, self.hyperparameters)
    levels = np.linspace(self.mses_ridge.min(),self.mses_ridge.max(), 7)
    contour = ax.contourf(
      degrees_mesh, hyperparameters_mesh, self.mses_ridge.T, levels=levels)
    ax.set_yscale("log")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Ridge parameter")
    ax.grid()
    format_func = lambda x, _: f"{x:.2f}"
    cbar = plt.colorbar(contour, format=format_func)
    fig.tight_layout()
    fig.savefig("ridge_mse.pdf")