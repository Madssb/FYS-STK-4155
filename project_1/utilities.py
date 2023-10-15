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
import re

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


def convert_to_label(metric_name):
    """
    Replaces "mean_squared_error" with "MSE" and "r2_score" with "R2-score" in a given metric name.
    """
    # Use regular expressions to search for the function names in the metric name
    mse_match = re.search(r"mean_squared_error", metric_name)
    r2_score_match = re.search(r"r2_score", metric_name)
    ols_match = re.search(r"ols", metric_name)
    ridge_match = re.search(r"ridge", metric_name)
    lasso_match = re.search(r"lasso", metric_name)

    # Replace the function names with the desired format
    if mse_match:
        metric_name = re.sub(r"mean_squared_error", "MSE", metric_name)
    if r2_score_match:
        metric_name = re.sub(r"r2_score", r"R$^2$ score", metric_name)
    if ols_match:
       metric_name = re.sub(r"ols", "OLS", metric_name)
    if ridge_match:
       metric_name = re.sub(r"ridge", "Ridge", metric_name)
    if lasso_match:
       metric_name = re.sub(r"lasso", "Lasso", metric_name)
    return metric_name


if __name__ == '__main__':
   pass
