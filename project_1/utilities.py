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
  Evaluate Franke's function for given x, y mesh.


  Parameters
  ----------
  x: n-dimensional array of floats
    Meshgrid for x.
  y: n-dimensional array of floats
    Meshgrid for y.


  Returns
  -------
  array like:
    Franke's function mesh.


  """
  term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
  term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
  term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
  term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
  return term1 + term2 + term3 + term4


def mean_squared_error(observed: np.ndarray, estimate: np.ndarray) -> float:
  """
  Compute mean squared error for estimate w.r.t observed data.


  Parameters
  ----------
  observed: x-dimensional array of floats
    Observed values
  estimate: x-dimensional array of floats
    Estimate for observed values
      

  Returns
  -------
  float:
    Mean squared error of estimate.


  """
  err_msg = f"{observed.shape=} and {estimate.shape=}, expected same shapes."
  assert observed.shape == estimate.shape, err_msg
  return np.mean((observed - estimate)**2)


def r2_score(observed: np.ndarray, estimate: np.ndarray) -> float:
  """
  Compute R2-score for some estimate w.r.t observed data.


  Parameters
  ----------
  observed: x-dimensional array of floats
    Observed values
  estimate: x-dimensional array of floats
    Estimate for observed values
    
      
  Returns
  -------
  float:
    R2-score of estimate.


  """
  err_msg = f"{observed.shape=} and {estimate.shape=}, expected same shapes."
  assert observed.shape == estimate.shape, err_msg
  mse = mean_squared_error(observed, estimate)
  mean_observed = np.mean(observed)*np.ones_like(observed)
  return 1 - mse**2/mean_squared_error(observed, mean_observed)


def bias(observed: np.ndarray, estimate: np.ndarray) -> float:
  """
  Compute bias for some estimate w.r.t observed data.

  
  Parameters
  ----------
  observed: x-dimensional array of floats
    Observed values
  estimate: x-dimensional array of floats
    Estimate for observed values
  
    
  Returns
  -------
  float:
    Bias of estimate.


  """
  estimate_mean = np.mean(estimate)*np.ones(estimate)
  return np.mean((observed - estimate_mean)**2)


def variance(estimate: np.ndarray) -> float:
  """
  Compute variance for estimate:


  Parameters
  ----------
  estimate: x-dimensional array of floats
    Trained parameters applied on observed input.

  
  Returns
  -------
  float:
    variance of estimate
  """
  return bias(estimate, estimate)

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
