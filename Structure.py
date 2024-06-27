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
        self._trainingData = {}
        
        #Initalise all the layers
        self._layers = {}
        self._layers[0] = Layer(layerSizes[0], None)
        for i in range(1, self._depth):
            self._layers[i] = Layer(layerSizes[i], self._layers[i-1])
            
    def first_layer(self):
        return self._layers[0]
    
    def last_layer(self):
        return self._layers[self._depth-1]
    
    def get_network_weights_biases(self) -> list[tuple[np.matrix, np.ndarray]]:
        layers = self._layers
        return [(layers[i].get_weights(), layers[i].get_biases()) for i in range(self._depth)]
        
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
    
    def set_training_data_paths(self, directory: str, labels:list) -> None:
        """Assigns each data file's path to it's label by setting the trainingData parameter. 
            Currently only works for pngs

        Args:
            directory (str): The relative directory to obtain the training data. This method 
            expects the data to be organised such that each piece of data with the same label
                is in it's own folder, where the folder is named what the label is.
            labels (list[Any]): the list of labels for the data. These should match the names of
                the folders in the directory. Order of these does not need to match order of folders.
                
        Returns: 
            None 
        """
        #Store and sorts all the directories to their label
        self._trainingData
        for label in labels:
            self._trainingData[label] = collect_images(f"{directory}/{label}")
            
    def train_network():
        return NotImplemented        
        
#Test
test_network = Network([784,16,16,10])
test_network.set_training_data_paths("../24CookieTrainingData/sample/dataset", [0,1,2,3,4,5,6,7,8,9])
print((test_network._trainingData))
            
            
        