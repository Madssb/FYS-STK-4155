"""
This model contains the regression functions pertaining to project 1.
"""
import numpy as np


def design_matrix_polynomial_xy(x: np.ndarray, y: np.ndarray, degree: int,
                                scale=True)-> np.ndarray:
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
  design_matrix_x = np.empty((len(x),degree+1))
  design_matrix_y = np.empty_like(design_matrix_x)
  for index in range(degree+1):
      design_matrix_x[:, index] = x**index
      design_matrix_y[:, index] = y**index
  design_matrix_xy = np.empty((len(x),(degree+1)**2))
  for row in range(len(x)):
    for i, xn_pow in enumerate(design_matrix_x[row,:]):
      for j, yn_pow in enumerate(design_matrix_y[row, :]):
        col = i*(degree + 1) + j
        design_matrix_xy[row, col] = xn_pow*yn_pow
  if scale:
    design_matrix_xy -= np.mean(design_matrix_xy, axis=1, keepdims=True)
  return design_matrix_xy


def test_design_matrix_polynomial_xy():
  """ 
  Ensures design_matrix_polynomial_xy() is working as intended.
  """
  x = np.array([2, 3])
  y = np.array([4, 5])
  expected_design_matrix = np.array([[1, 4, 16, 2, 8, 32, 4, 16, 64],
                                     [1, 5, 25, 3, 15, 75, 9, 45 ,225]])
  design_matrix = design_matrix_polynomial_xy(x,y,2,scale=False)
  assert design_matrix.shape == (2,9)
  print(design_matrix)
  assert (design_matrix == expected_design_matrix).all()


def ols_regression(design_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
  """
  Compute the optimal parameters per Ordinary Least Squares regression.


  Parameters
  ----------
  design_matrix
    Two dimensional numpy array
  y: one-dimensional array of floats
    Y-dimension mesh.
  
  
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
