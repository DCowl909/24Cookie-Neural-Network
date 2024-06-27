import math
import numpy as np


def sig(x: float) -> float:
    """ This function implements the sigmoid function for a scalar"""
    C = math.e
    return 1/(1+C**(-x))\



def sigmoid(z: np.ndarray) -> np.ndarray:
    """ This function implements the sigmoid function, and 
    expects a numpy array as argument """
    
    if not isinstance(z, np.ndarray):
        raise ValueError("The input must be a numpy array")
    
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return sigmoid

def d_sig(x: float) -> float:
    """ This function implements the derivative of the sigmoid function for a scalar"""
    C = math.e
    return C**(-x) * (sig(x))**2 

def d_sigmoid(z: np.ndarray) -> np.ndarray:
    """ This function implements the derivative of the sigmoid function, and 
    expects a numpy array as argument """
    
    if not isinstance(z, np.ndarray):
        raise ValueError("The input must be a numpy array")
    
    d_sigmoid = np.multiply(np.exp(-z), np.square(sigmoid(z)))
    return d_sigmoid
    