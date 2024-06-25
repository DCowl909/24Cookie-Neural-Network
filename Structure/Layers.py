import numpy as np

"""
Represents an individual layer 

"""
class Layer:
    
    def __init__(self, size, previousLayer, nextLayer):
        
        self._size = size
        self._prevLayer = previousLayer
        self._nextLayer = nextLayer
        
        # Construct default weight matrix
        if previousLayer == None:
            self._weights = None
        else:
            prevLayerSize = previousLayer.size()
            self._weights = np.ones((prevLayerSize, prevLayerSize))

    """
    Returns the size of this layer (the number of neurons)
    """
    def size(self) -> int:
        return self._size
    
    def print_weights(self) -> str:
        return str(self._weights)
    
test_layer = Layer(19, None, "chicekn")
print(test_layer.print_weights())