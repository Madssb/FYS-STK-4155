import numpy as np
import sklearn.metrics as skm
"""
Contains functions that compute model evaluation metrics.
"""

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
  #mse = mean_squared_error(observed, estimate)
  #mean_observed = np.mean(observed)*np.ones_like(observed)
  #return 1 - np.sum((observed - estimate)**2)**2/mean_squared_error(observed, mean_observed)
  SSE = np.sum((observed - estimate)**2)
  Var = np.sum((observed - np.mean(observed))**2)
  return 1 - SSE/Var

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
  estimate_mean = np.mean(estimate, axis=1, keepdims=True)
  return np.mean((observed - estimate_mean)**2 )


def variance(estimate: np.ndarray) -> float:
  """
  Compute variance for estimate:


  Parameters
  ----------
  estimate: x-dimensional array of floats
    Trained parameters applied on observed input.
(
  
  Returns
  -------
  float:
    variance of estimate
  """

  return np.mean(np.var(estimate, axis=1, keepdims=True)) 

def mean_squared_error_bootstrapped(observed: np.ndarray, estimate: np.ndarray) -> float:

  mse = np.mean(np.mean((observed - estimate)**2, axis=1, keepdims=True))
  return mse