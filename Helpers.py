import math
import numpy as np


def sig(x: float) -> float:
    """ This function implements the sigmoid function for a scalar"""
    C = math.e
    return 1/(1+C**(-x))\

def inv_sig(x: float) -> float:
    """This function implements the inverse sigmoid function for a scalar"""
    if (x == 1):
        return ValueError("1 in the inverse sigmoid function fuck off")
    
    return -1 * np.log((1/x) - 1)

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
    
def maxNeuron(neuronArr: np.ndarray):
    """Returns the index of an array that has the greatest value in that array."""
    size = neuronArr.size
    return max((neuronArr[i], i) for i in range(size))[1]


print(sig(69))
inv_sig(1)