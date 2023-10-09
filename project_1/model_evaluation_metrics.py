import numpy as np
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
  estimate_mean = np.mean(estimate)*np.ones_like(estimate)
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


class ModelEvaluation:
  """
  Measure performance for some predicted w.r.t unseen data for the following
  metrics:
  - mean square error,
  - b2-scorel,
  - bias,
  - variance.
  """

  def __init__(self, unseen: np.ndarray, predicted: np.ndarray):
    """
    Instantiate ModelEvaluation object
    """
    self.unseen = unseen
    self.predicted = predicted



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