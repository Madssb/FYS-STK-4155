"""
This module contains several utility functions pertaining to project 1
in FYS-STK4155.

Functions:
- franke_function: Evaluates the franke_function for some mesh x,y.
- mean_squared_error: Compute the MSE for some model and some corresponding
  analytical expression.
- r2_score: Compute the R2-score for some model and some corresponding
  analytical expression.
"""
import numpy as np


def franke_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """
  Evaluates the franke function for coordinates x and y.


  Parameters
  ----------
  x: Two-dimensional array of floats
    Meshgrid for x.
  y: Two-dimensional array of floats
    Meshgrid for y.


  Returns
  -------
  Two-dimensional array of floats
    FrankeFunction evaluated on x, y mesh.
  """
  term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
  term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
  term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
  term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
  return term1 + term2 + term3 + term4


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


def test_mean_squared_error_5_dim():
  """
  Checks validity of mse function for larger dimensions, ensures functionality
  is as expected.
  """
  a = np.ones((3, 3, 3, 3, 3))
  b = a*1.1
  try:
    mse = mean_squared_error(a, b)
    print(mse)
    assert True
  except:
     assert False


def test_r2_score():
  """
  Checks validity of mse function for larger dimensions, ensures functionality
  is as expected.
  """
  a = np.ones((3, 3, 3, 3, 3))
  b = a*1.1
  try:
    r2_score(a, b)
    assert True
  except:
    assert False


def my_figsize(column=True, subplots=(1, 1), ratio=None):
  """
  Specifies figure dimensions best suitable for latex.
  Credit to Johan Carlsen.
  """
  if column: 
    width_pt = 255.46837
  else:
    #width of latex text
    width_pt = 528.93675
  inch_per_pt = 1/72.27
  fig_ratio = (5**0.5 - 1)/2
  fig_width = width_pt*inch_per_pt
  fig_height = fig_width*fig_ratio*subplots[0]/subplots[1]
  fig_dim = (fig_width,fig_height)
  return fig_dim


if __name__ == '__main__':
   pass
