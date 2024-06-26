import numpy as np
from Helpers import *

"""
Represents an individual layer 

"""
class Layer:
    
    def __init__(self, size: int, previousLayer):
        
        self._size = size
        self._prevLayer = previousLayer
        
        #Initialise neurons
        self._neurons = np.zeros(size).reshape(-1, 1)
        
        # If this is starting layer
        if previousLayer == None:
            self._weights = None
            self._biases = None
        else:
            # Initalise weights and biases
            prevLayerSize = previousLayer.size()
            #self._weights = np.zeros((size, prevLayerSize))
            self._weights = np.random.uniform(-1, 1, (size, prevLayerSize))
            print(self._weights)
            self._biases = np.zeros(size).reshape(-1, 1)
            

    def execute(self) -> None:
        prev_neurons = self._prevLayer.get_neurons()
        calculatedNeurons = np.dot(self._weights, prev_neurons) + self._biases
        self._neurons = sigmoid(calculatedNeurons)
        
    def size(self) -> int:
        return self._size
    
    def get_neurons(self):
        return self._neurons
    
    def set_neurons(self, neurons:list[float]):
        if isinstance(neurons, list):
            self._neurons = np.array(neurons).reshape(-1, 1)
        elif isinstance(neurons, np.ndarray):
            if neurons.shape[1]== 1: #if it is a column vector
                self._neurons = neurons
            elif neurons.shape[0] == 1: #if it is a row vector
                self._neurons = neurons.reshape(-1, 1)
            else:
                raise ValueError("Unsupported neuron data format")
        else:
            raise ValueError("Unsupported neuron data format")
                
        
    def get_weights(self):
        return self._weights
    
    def get_biases(self):
        return self._biases
    
    def toString(self):
        return str(self._neurons)


"""
Represents the entire network which manages a bunch of layers.

"""
class Network:
    
    def __init__(self, layerSizes: list[int]):
        
        self._depth = len(layerSizes)
        
        #Initalise all the layers
        self._layers = {}
        self._layers[0] = Layer(layerSizes[0], None)
        for i in range(1, self._depth):
            self._layers[i] = Layer(layerSizes[i], self._layers[i-1])
            
    def first_layer(self):
        return self._layers[0]
    
    def last_layer(self):
        return self._layers[self._depth-1]
            
    def get_network_layers(self) -> dict[int, Layer]:
        return self._layers
    
    def execute_network(self, input_data: np.ndarray | list):
        if len(input_data) != self._layers[0].size():
            raise ValueError("Input data has dimension not matching first neural layer.")
        
        self.first_layer().set_neurons(input_data)
        for i in range(1, self._depth):
            self._layers[i].execute()
            
        print("LAST LAYER:")
        print(self._layers[i].get_neurons())
        return np.max(self.last_layer().get_neurons())
            
        