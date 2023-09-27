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