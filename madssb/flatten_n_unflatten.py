import numpy as np

def flatten(parameters):
    """
    Convert tuple of ndarrays and integers into a flattened tuple of ndarrays and integers.
    
    Parameters
    ----------
    parameters: tuple
        Parameters to flatten

    Returns
    -------
    tuple
        Flattened parameters, unflatten callable
    """
    shapes = [param.shape if isinstance(param, np.ndarray) else None for param in parameters]
    sizes = [param.size if isinstance(param, np.ndarray) else 1 for param in parameters]
    parameters_flattened = np.concatenate([param.ravel() if isinstance(param, np.ndarray) else np.array([param]) for param in parameters])
    
    def undo_func(flattened_params):
        params = []
        start = 0
        for shape, size in zip(shapes, sizes):
            end = start + size
            if shape is not None:
                param = flattened_params[start:end].reshape(shape)
            else:
                param = flattened_params[start:end][0]
            params.append(param)
            start = end
        return tuple(params)
    
    return parameters_flattened, undo_func

# Test the flatten and undo_func functions
w_1 = np.ones((2, 2))
w_out = np.ones((2, 1))
b_1 = np.ones((2, 1))
b_out = 1
parameters = (w_1, w_out, b_1, b_out)

parameters_flattened, undo_func = flatten(parameters)
print("Flattened Parameters:")
print(parameters_flattened)
print("\nOriginal Parameters:")
print(parameters)

# Test undo_func
restored_parameters = undo_func(parameters_flattened)
print("\nRestored Parameters:")
print(restored_parameters)
