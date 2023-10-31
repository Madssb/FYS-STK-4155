import numpy as np

def mean_squared_error(target: np.ndarray, output: np.ndarray) -> np.ndarray:
  return np.mean((target - output)**2)
