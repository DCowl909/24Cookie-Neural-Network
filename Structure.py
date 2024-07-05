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
        
        # If this is starting layer
        if previousLayer == None:
            self._weights = None
            self._biases = None
        else:
            # Initalise weights and biases
            prevLayerSize = previousLayer.size()
            self._weights = np.random.uniform(-1, 1, (size, prevLayerSize))
            self._biases = np.zeros(size).reshape(-1, 1)
            

    def execute(self) -> None: 
        #performs the matrix multiplication to get the values for this layers neurons
        prev_neurons = self._prevLayer.get_neurons()
        calculatedNeurons = np.dot(self._weights, prev_neurons) + self._biases
        self._neurons = sigmoid(calculatedNeurons)
        
    def size(self) -> int:
        return self._size
    
    def get_neurons(self):
        return self._neurons
    
    def get_neuron(self, n):
        return self._neurons[n]
    
    def set_neurons(self, neurons:list[float]):
        if isinstance(neurons, list):
            self._neurons = np.array(neurons).reshape(-1, 1)
        elif isinstance(neurons, np.ndarray):
            if len(neurons.shape) == 1: #if it is a row vector
                self._neurons = neurons.reshape(-1, 1)
            elif neurons.shape[1]== 1: #if it is a column vector
                self._neurons = neurons
            else:
                raise ValueError("Unsupported neuron data format")
        else:
            raise ValueError("Unsupported neuron data format")
                
    def get_weights(self):
        return self._weights
    
    def change_weights(self, weights: np.matrix):
        self._weights = self._weights + weights
        
    def set_weights(self, weights: np.matrix):
        self._weights = weights
        
    def get_biases(self):
        return self._biases
    
    def set_biases(self, biases: np.ndarray):
        self._biases = biases
        
    def change_biases(self, biases: np.ndarray):
        self._biases = self._biases + biases

"""
Represents the entire network which manages a bunch of layers.

"""
class Network:
    
    def __init__(self, layerSizes: list[int]):
        
        self._depth = len(layerSizes)
        self._trainingData = {}
        self._layerSizes = layerSizes
        
        #Initalise all the layers
        self._layers = {}
        self._layers[0] = Layer(layerSizes[0], None)
        for i in range(1, self._depth):
            self._layers[i] = Layer(layerSizes[i], self._layers[i-1])
            
    def get_depth(self):
        return self._depth
    
    def duplicate(self):
        """Returns a network with the same structure to this one"""
        return Network(self._layerSizes)
    
    def get_layer_neurons(self, layer: int) -> np.ndarray:
        return self._layers[layer].get_neurons()
    
    def set_layer_neurons(self, layer: int, neurons: np.ndarray) -> None:
        self._layers[layer].set_neurons(neurons)
        
    def get_network_weights_biases(self) -> list[tuple[np.matrix, np.ndarray]]:
        layers = self._layers
        return [(layers[i].get_weights(), layers[i].get_biases()) for i in range(1, self._depth)]
    
    def set_network_weights_biases(self, weightsBiases: list[tuple[np.matrix, np.ndarray]]):
        for i in range(self._depth - 1):
            currentLayer = self._layers[i+1] #as input weights and biases start from 2nd layer
            currentLayer.set_weights(weightsBiases[i][0])
            currentLayer.set_biases(weightsBiases[i][1]) 
            
    def change_network_weights_biases(self, weightsBiases: list[tuple[np.matrix, np.ndarray]]):
        for i in range(self._depth - 1):
            currentLayer = self._layers[i+1] #as input weights and biases start from 2nd layer
            currentLayer.change_weights(weightsBiases[i][0])
            currentLayer.change_biases(weightsBiases[i][1])
            
        
    def get_network_layers(self) -> dict[int, Layer]:
        return self._layers
    
    def get_network_layer(self, n: int) -> Layer:
        return self._layers[n]
    
    def get_last_layer(self) -> Layer:
        return self._layers[self._depth-1]
    
    def execute_network(self, input_data: np.ndarray | list) -> np.ndarray:
        """Executes the network with an input vector. Alters the state of the network, and returns the last layer"""
        firstLayer = self._layers[0]
        lastLayer = self._layers[self._depth-1]
        if len(input_data) != firstLayer.size():
            raise ValueError("Input data has dimension not matching first neural layer.")
        
        firstLayer.set_neurons(input_data)
        for i in range(1, self._depth):
            self._layers[i].execute()
        
        return lastLayer.get_neurons()
    
    def evaluate(self, input_data: np.ndarray | list) -> int:
        outputVector = self.execute_network(input_data)
        return max((outputVector[i],i) for i in range(len(outputVector)))[1]
    
    def save_network(self, name: str):
        # Create a list of arrays
        arrays = [np.array(self._layerSizes)] #network structure info
        arrays += [self._layers[l].get_weights() for l in range(1, self._depth)] #weights
        arrays += [self._layers[l].get_biases() for l in range(1, self._depth)] #biases

        np.savez(f"TrainedNetworks/{name}.npz", **{f'array_{i}': arr for i, arr in enumerate(arrays)})

    def load_network(self, name: str):
    
        data = np.load(f"TrainedNetworks/{name}.npz", allow_pickle=True)
        arrays_loaded = [data[f'array_{i}'] for i in range(len(data.files))]
        
        Structure = arrays_loaded[0]
        weights = arrays_loaded[1 : self._depth]
        biases = arrays_loaded[self._depth : 2*self._depth]
        for l in range(1, self._depth):
            self._layers[l].set_weights(weights[l-1])
            self._layers[l].set_biases(biases[l-1])
            
            
        
    
        
        
        
    