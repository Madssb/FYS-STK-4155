import jax.numpy as jnp

def sigmoid(activation):
    """Compute out for some neuron.

    Parameters
    ----------
    activation: np.ndarray
      Activation.
    
    Returns
    -------
      np.ndarray
    """
    return 1 / (1 + jnp.exp(-activation))