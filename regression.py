"""
This model contains the regression functions pertaining to project 1.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utilities import my_figsize

def design_matrix_polynomial_xy(x: np.ndarray, y: np.ndarray, degree: int,
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
  design_matrix_xy = np.empty((len_x*len_y, (degree+1)**2), dtype=float)
  for i, x_ in enumerate(x):
    for j, y_ in enumerate(y):
      row = len_y*i + j
      for k in range(degree + 1):
        for l in range(degree + 1):
          col = k*(degree+1) + l
          design_matrix_xy[row, col] = x_**k*y_**l
  if scale:
    design_matrix_xy -= np.mean(design_matrix_xy, axis=1, keepdims=True)
  return design_matrix_xy


def test_design_matrix_polynomial_xy():
  """ 
  Ensures design_matrix_polynomial_xy() is working as intended.
  """
  x = np.array([2, 3])
  y = np.array([4, 5])
  expected_design_matrix = np.array([[1, 4, 2, 8],
                                     [1, 5, 2, 10],
                                     [1, 4, 3, 12],
                                     [1, 5, 3, 15]])
  design_matrix = design_matrix_polynomial_xy(x, y, 1, scale=False)
  assert design_matrix.shape == (4, 4)
  print(design_matrix)
  assert (design_matrix == expected_design_matrix).all()


def ols_regression(design_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
  """
  Compute the optimal parameters per Ordinary Least Squares regression.


  Parameters
  ----------
  design_matrix: two-dimensional array of floats
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
      shapes of design_matrix or y are not permitted.

    
  """
  assert len(design_matrix.shape) == 2, "requires nxm dimensional array."
  assert len(y.shape) == 1, "requires n dimensional array."
  return np.linalg.pinv(
      np.transpose(design_matrix) @ design_matrix
  ) @ np.transpose(design_matrix) @ y


def ridge_regression(design_matrix: np.ndarray, y: np.ndarray,
                     hyperparameter: float) -> np.ndarray:
  """
  Computes the optimal parameters per Ridge regression.

  Parameters
  ----------
  design_matrix
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
    shapes of design_matrix or y are not permitted, or hyperparameter is not
    float.


  """
  assert len(design_matrix.shape) == 2, "requires nxm dimensional array."
  assert len(y.shape) == 1, "requires n dimensional array"
  assert isinstance(hyperparameter, float), "must be float"
  return np.linalg.pinv(
      np.transpose(design_matrix) @ design_matrix
      + np.identity(design_matrix.shape[1])*hyperparameter
  ) @ np.transpose(design_matrix) @ y




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

  def design_matrix_polynomial_xy(
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
    design_matrix_xy = np.empty((len_x*len_y, (degree+1)**2), dtype=float)
    for i, x_ in enumerate(x):
      for j, y_ in enumerate(y):
        row = len_y*i + j
        for k in range(degree + 1):
          for l in range(degree + 1):
            col = k*(degree+1) + l
            design_matrix_xy[row, col] = x_**k*y_**l
    if self.scale:
      design_matrix_xy -= np.mean(design_matrix_xy, axis=1, keepdims=True)
    return design_matrix_xy

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
    design_matrix = self.design_matrix_polynomial_xy(self.x, self.y, degree)
    design_matrix_train, design_matrix_test, z_train, z_test = train_test_split(design_matrix, self.z_flat)
    optimal_parameters = np.linalg.pinv(
        np.transpose(design_matrix_train) @ design_matrix_train
    ) @ np.transpose(design_matrix_train) @ self.z_train
    predicted_model = design_matrix_test @ optimal_parameters
    return predicted_model

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
    design_matrix_train = self.design_matrix_polynomial_xy(
      self.x_train, self.y_train, degree)
    design_matrix_test = self.design_matrix_polynomial_xy(
      self.x_test, self.y_test, degree)
    optimal_parameters =  np.linalg.pinv(
        np.transpose(design_matrix_train) @ design_matrix_train
        + np.identity(design_matrix_train.shape[1])*hyperparameter
    ) @ np.transpose(design_matrix_train) @ self.z_train
    predicted_model = design_matrix_test @ optimal_parameters
    return predicted_model
  
  def mse(self, regression_type: str):
    """
    Compute the mean squared error for OLS- Ridge- or Lasso regression, as a 
    function of polynomial degrees, and additionally the hyperparameter if the
    regression type is Ridge. Visualizes result and


    Parameters
    ----------
    regression_type: str
      type of regression applied to data.
    """
    regression_type = regression_type.lower()
    err_msg = f"{regression_type} is not ridge, lasso, or ols"
    assert regression_type.lower() in ["ridge", "lasso", "ols"], err_msg 
    if regression_type == "ols":
      self.mses_ols = np.empty_like(self.degrees,dtype=float)
      for i, degree in enumerate(self.degrees):
        model = self.ols_regression(degree)
        self.mses_ols[i] = np.mean((self.z_test - model)**2)
      fig, ax = plt.subplots(figsize=my_figsize())
      ax.plot(self.degrees, self.mses_ols, label="MSE")
      ax.set_xlabel("Complexity")
      ax.set_ylabel("MSE")
      fig.savefig("mse_ols.pdf")
    elif regression_type == "ridge":
      self.mses_ridge = np.empty(
        self.degrees.shape[0], self.hyperparameters.shape[0], dtype=float)
      for i, degree in enumerate(self.degrees):
        for j, hyperparameter in enumerate(self.hyperparameters):
          model = self.ridge_regression(degree, hyperparameter)
          self.mses_ridge[i][j] = np.mean((self.z_test - model)**2)
      #visualize results
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

  def mse_ols(self, visualize=True):
    """
    Compute mean squared error for ordinary least square regression computed
    models as a function of model complexity.
    """
    self.mses_ols = np.empty_like(self.degrees,dtype=float)
    for i, degree in enumerate(self.degrees):
      model = self.ols_regression(degree)
      self.mses_ols[i] = np.mean((self.z_test - model)**2)
    if not visualize:
      return
    fig, ax = plt.subplots(figsize=my_figsize())
    ax.plot(self.degrees, self.mses_ols, label="MSE")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("MSE")
    fig.savefig("mse_ols.pdf")  

  def mse_ridge(self, visualize=True):
    """
    Compute mean squared error for Ridge regression computed models as a
    function of model complexity.
    """
    self.mses_ridge = np.empty(
      self.degrees.shape[0], self.hyperparameters.shape[0], dtype=float)
    for i, degree in enumerate(self.degrees):
      for j, hyperparameter in enumerate(self.hyperparameters):
        model = self.ridge_regression(degree, hyperparameter)
        self.mses_ridge[i][j] = np.mean((self.z_test - model)**2)
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