import math
import numpy as np


def sig(x: float) -> float:
    """ This function implements the sigmoid function for a scalar"""
    C = math.e
    return 1/(1+C**(-x))\



def sigmoid(z: np.array) -> np.array:
    """ This function implements the sigmoid function, and 
    expects a numpy array as argument """
    
    if not isinstance(z, np.ndarray):
        raise ValueError("The input must be a numpy array")
    
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return sigmoid
