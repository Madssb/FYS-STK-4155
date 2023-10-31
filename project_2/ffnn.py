"""

"""
import numpy as np

class FeedForwardNeuralNetwork:
  """Feed Forward Neural Network
  """

  def __init__(self, n_inputs: int, n_hidden_layers: int,
               n_neurons: int, n_outputs: int,
               activation_function: callable):
    """Constructor

    Parameters
    ----------
    n_inputs : int
      Number of inputs (expected to be an integer).
    n_hidden_layers : int
      Number of hidden layers (expected to be an integer).
    n_neurons : int
      Number of neurons per hidden layer (expected to be an integer).
    n_outputs : int
      Number of nodes in output (expected to be an integer).
    activation_function : callable
      Activation function (expected to be a callable function).

    Raises
    ------
    TypeError
        If any of the input parameters does not have the expected type.
    """
    # Check types of input parameters
    if not isinstance(n_inputs, int):
        raise TypeError("n_inputs must be an integer.")
    if not isinstance(n_hidden_layers, int):
        raise TypeError("n_hidden_layers must be an integer.")
    if not isinstance(n_neurons, int):
        raise TypeError("n_neurons must be an integer.")
    if not isinstance(n_outputs, int):
        raise TypeError("n_outputs must be an integer.")
    if not callable(activation_function):
        raise TypeError("activation_function must be a callable function.")
    # Initialize instance variables
    self.n_inputs = n_inputs
    self.n_hidden_layers = n_hidden_layers
    self.n_neurons = n_neurons
    self.n_outputs = n_outputs
    self.activation_function = activation_function

  def initialize_weights_and_biases(self):
    """Initialize weights and biases with normal dist.

    """
    self.weights = []
    self.biases = []
    for _ in range(self.n_hidden_layers):
      weight = np.random.normal(size=(self.n_neurons, self.n_inputs))
      bias = np.random.normal(size=(self.n_neurons, 1))
      self.weights.append(weight)
      self.biases.append(bias)
    weight = np.random.normal(size=(self.n_neurons, 1))
    bias = np.random.normal(size=(self.n_outputs, 1))
    self.weights.append(weight)
    self.biases.append(bias)


  def extract_sparameters(self):
    """Extract model parameters.

    Returns
    -------
    tuple
      Parameters
    """
    assert all(isinstance(weight, np.ndarray) for weight in self.weights), "Not all elements in self.weights are NumPy ndarrays"
    assert all(isinstance(bias, np.ndarray) for bias in self.biases), "Not all elements in self.biases are NumPy ndarrays"
    parameters = (np.concatenate([w.ravel() for w in self.weights]),
                  np.concatenate([b.ravel() for b in self.biases]))
    return parameters
  
  def insert_parameters(self, parameters):
    """
    Do the opposite of `weights_and_biases_to_parameters`.
    """
    weight_shapes = [w.shape for w in self.weights]
    bias_shapes = [b.shape for b in self.biases]
    weight_sizes = [np.prod(shape) for shape in weight_shapes]
    bias_sizes = [np.prod(shape) for shape in bias_shapes]

    weight_indices = np.cumsum([0] + weight_sizes)
    bias_indices = np.cumsum([0] + bias_sizes)

    self.weights = [np.reshape(parameters[0][start:end], shape)
                    for start, end, shape in zip(weight_indices[:-1], weight_indices[1:], weight_shapes)]
    self.biases = [np.reshape(parameters[1][start:end], shape)
                   for start, end, shape in zip(bias_indices[:-1], bias_indices[1:], bias_shapes)]

  def __call__(self, features: np.ndarray):
    """Compute forward pass for FFNN

    Parameters
    ----------
    features: np.ndarray
      Features.

    Returns
    -------
      float, shape: (n_outputs)
    """
    input = features
    for i, weight in enumerate(self.weights): 
      activation = weight.T @ input + self.biases[i]
      output = self.activation_function(activation)   
      input = output
    return output

  def meta_forward_pass(self, parameters: tuple,
                        features: np.ndarray):
    """Compute forward pass for FFNN w.r.t parameters

    Parameters
    ----------
    features: np.ndarray
      Features.
    
    Returns
    -------
      float, shape: (n_outputs)
    """
    self.insert_parameters(parameters)
    return self.__call__(features)

