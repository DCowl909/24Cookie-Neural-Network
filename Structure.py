#should put in some defensive programming

import numpy as np
from Helpers import *
from ImageProcessing import *
import glob

"""
Represents an individual layer 

"""
class Layer:
    
    def __init__(self, size: int, previousLayer):
        
        self._size = size
        self._prevLayer = previousLayer
        
        #Initialise neurons
        self._neurons = np.zeros(size).reshape(-1, 1)
        self._unstandardisedNeurons = self._neurons #used for the purpose of backpropogation
        
        # If this is starting layer
        if previousLayer == None:
            self._weights = None
            self._biases = None
        else:
            # Initalise weights and biases
            prevLayerSize = previousLayer.size()
            self._weights = np.random.uniform(-5, 5, (size, prevLayerSize))
            self._biases = np.random.uniform(-5, 5, size).reshape(-1,1)
            

    def execute(self) -> None: 
        """performs the matrix multiplication to set the values for this layer's neurons"""
        prev_neurons = self._prevLayer._neurons
        calculatedNeurons = np.dot(self._weights, prev_neurons) + self._biases
        self._unstandardisedNeurons = calculatedNeurons
        self._neurons = sigmoid(calculatedNeurons)
        
    def size(self) -> int:
        return self._size
    
    def change_weights(self, weights: np.matrix):
        self._weights = self._weights + weights
        
    def change_biases(self, biases: np.ndarray):
        self._biases = self._biases + biases


"""
Represents the entire network which manages a bunch of layers.

"""
class Network:
    
    def __init__(self, layerSizes: list[int]):
        
        self._depth = len(layerSizes)
        self._complexity = sum(layerSizes[i] + layerSizes[i-1]*layerSizes[i] for i in range(1, self._depth)) #how many weights and biases there are
        self._structure = layerSizes
        
        #Initalise all the layers
        self._layers = []
        self._layers += [Layer(layerSizes[0], None)]
        for i in range(1, self._depth):
            self._layers += [Layer(layerSizes[i], self._layers[i-1])]
            
    def get_depth(self):
        return self._depth
    
    def get_layer_size(self, layer):
        return self._layers[layer].size()
    
    def get_complexity(self):
        return self._complexity
    
    def duplicate(self):
        """Returns a network with the same structure to this one"""
        return Network(self._structure)
    
    def get_neurons(self, layer: int) -> np.ndarray:
        """Returns the array of neurons in the network at a specified layer."""
        return self._layers[layer]._neurons
    
    def set_neurons(self, layer: int, neurons: np.ndarray) -> None:
        """Sets array of neurons in the network at a specified layer to a given array"""
        self._layers[layer]._neurons = neurons
        
    def get_unstandardised_neurons(self, layer:int):
        return self._layers[layer]._unstandardisedNeurons
        
    def get_neuron(self, layer: int, index: int) -> np.ndarray:
        """Returns the array of neurons in the network at a specified layer."""
        return self._layers[layer]._neurons[index]
    
    def set_neuron(self, layer: int, index : int, neuron: int) -> None:
        """Sets array of neurons in the network at a specified layer to a given array"""
        self._layers[layer]._neurons[index] = neuron
        
    def get_network_weights_biases(self) -> list[tuple[np.matrix, np.ndarray]]:
        layers = self._layers
        return [(layers[i]._weights, layers[i]._biases) for i in range(1, self._depth)]
    
    def get_layer_weights(self, layer):
        return self._layers[layer]._weights
    
    def get_layer_biases(self, layer):
        return self._layers[layer]._biases
    
    def set_network_weights_biases(self, weightsBiases: list[tuple[np.matrix, np.ndarray]]):
        for i in range(self._depth - 1):
            currentLayer = self._layers[i+1] #as input weights and biases start from 2nd layer
            currentLayer._weights = weightsBiases[i][0]
            currentLayer._biases = weightsBiases[i][1]
            
    def change_network_weights_biases(self, weightsBiases: list[tuple[np.matrix, np.ndarray]]):
        for i in range(self._depth - 1):
            currentLayer = self._layers[i+1] #as input weights and biases start from 2nd layer
            currentLayer.change_weights(weightsBiases[i][0])
            currentLayer.change_biases(weightsBiases[i][1])
    
    def execute_network(self, input_data: np.ndarray | list) -> np.ndarray:
        """Executes the network with an input vector. Alters the state of the network, and returns the last layer."""
        firstLayer = self._layers[0]
        lastLayer = self._layers[self._depth-1]
        
        if len(input_data) != firstLayer.size():
            raise ValueError("Input data has dimension not matching first neural layer.")
        
        firstLayer._neurons = input_data
        for i in range(1, self._depth):
            self._layers[i].execute()
        
        return lastLayer._neurons
    
    def evaluate(self, input_data: np.ndarray | list) -> int:
        """Determines what number the network thinks an input is. """
        outputVector = self.execute_network(input_data)
        return max((outputVector[i],i) for i in range(len(outputVector)))[1]
    
    def save_network(self, name: str):
        # Create a list of arrays
        arrays = [np.array(self._structure)] #network structure info
        arrays += [self._layers[l]._weights for l in range(1, self._depth)] #weights
        arrays += [self._layers[l]._biases for l in range(1, self._depth)] #biases

        np.savez(f"TrainedNetworks/{name}.npz", **{f'array_{i}': arr for i, arr in enumerate(arrays)})

    def load_network(self, name: str):
    
        data = np.load(f"TrainedNetworks/{name}.npz", allow_pickle=True)
        arrays_loaded = [data[f'array_{i}'] for i in range(len(data.files))]
        
        self._structure = arrays_loaded[0]
        weights = arrays_loaded[1 : self._depth]
        biases = arrays_loaded[self._depth : 2*self._depth]
        for l in range(1, self._depth):
            self._layers[l]._weights = weights[l-1]
            self._layers[l]._biases = biases[l-1]
            
            
    def get_network_layer(self, l):
        return self._layers[l]
