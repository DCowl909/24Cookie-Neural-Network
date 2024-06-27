from Structure import *
import numpy as np

def cost(expected: np.ndarray, actual: np.ndarray) -> float:
    """Returns the summed difference of square of two arrays. Will throw a value error if arrays
    are not the same shape
    """
    if expected.shape != actual.shape:
        raise ValueError(f"Arrays to compare for cost function are not of same shape: {expected.shape} vs {actual.shape}")
    differences = np.square(expected - actual)
    return np.sum(differences)

def expected_final_layer(digit):
    neurons = np.zeros((10, 1))
    neurons[digit] = 1.00
    return neurons
    
    
print(expected_final_layer(8))