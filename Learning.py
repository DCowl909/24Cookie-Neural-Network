#should be able to do the training operations on whole arrays

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

def expected_final_layer(digit) -> np.ndarray:
    neurons = np.zeros((10, 1))
    neurons[digit] = 1.00
    return neurons

def meanCost(network: Network, data: dict[any, list[np.ndarray]]):
    totalCost = 0
    #n = dataSize(data)
    for label in data:
        for vector in data[label]:
            expected = expected_final_layer(label)
            actual = network.execute_network(vector)
            totalCost += cost(expected, actual)
    return totalCost

def dataSize(data: dict[any, list[np.ndarray]]):
    counter = 0
    for label in data:
        counter += len(data[label])
    return counter

    

class NetworkTrainer():
    
    def __init__(self, network: Network, data: dict[any, list[np.ndarray]]):
        self._network = network
        self._networkDepth = network.get_depth()
        self._data = data
        self._dataSize = dataSize(data)
        
    def train_network(self) -> Network:
        iterate = 500
        for i in range(iterate):
            nablaCost = self.total_nabla_cost()
            self._network.change_network_weights_biases(nablaCost)
            
            if (i%20 == 0):
                print(f"Current total cost is {meanCost(self._network, self._data)}")
                print(f"Progress: {100*i/iterate}%")
            
        return self._network
            
        
    def total_nabla_cost(self) -> list[tuple[np.matrix, np.ndarray]]:
        
        """Uses the backpropogation algorithm to determine the current negated nabla vector of the cost function. 
            Only works with the final layer having 10 neurons. """
            
        summedNablaC = NetworkTrainer.get_zero_nabla(self._network)
        for label in self._data:
            for vector in self._data[label]:
                dataVectorNablaC = self.local_nabla_cost(vector, label)
                summedNablaC = NetworkTrainer.addNablas(summedNablaC, dataVectorNablaC)
                
        return NetworkTrainer.divideNabla(summedNablaC, self._dataSize/10)
                
        
        #iterate through each training example
        #for each example, calculate the local nabla cost
        #average all of these gradient vectors
        #return that shit
        
    def local_nabla_cost(self, dataVector: np.ndarray, label: int) -> list[tuple[np.matrix, np.ndarray]]:
        """Uses the backpropogation algorithm to determine the current downhill nabla vector of the cost function 
        with a SPECIFIC DATA VECTOR. This returns -nabla_C."""
        
        #start at the last layer, and calculate all the dCostdNeurons
        #then calculate weights
        #then calculate biases
        
        self._network.execute_network(dataVector)
        nablaNetwork = self._network.duplicate() 
        
        #calculate all of the dCostdNeurons
        for l in range(self._networkDepth- 1, 0 , -1):
            if l == self._networkDepth - 1:
                dCdaL = 2 * (self._network.get_layer_neurons(l) - expected_final_layer(label)) #vector of dCostdNeurons
                
            else:
                currentLayer = self._network.get_network_layer(l)
                nextLayer = self._network.get_network_layer(l+1)
                
                dCdaL = np.zeros(currentLayer.size()).reshape(-1, 1)
                #j is nuerons in layer l + 1
                for j in range(nextLayer.size()):
                    w = nextLayer.get_weights()[j].reshape(-1,1)
                    daLplus1daL = w * d_sig(inv_sig(nextLayer.get_neuron(j)))
                    dCdaL += nablaNetwork.get_layer_neurons(l+1)[j] * daLplus1daL
        
            nablaNetwork.set_layer_neurons(l, dCdaL)
            
        #calculate all of the weights  and biases
        for l in range(self._networkDepth- 1, 0 , -1):
            
            aLminus1 = self._network.get_layer_neurons(l-1).reshape(-1)
            z = d_sigmoid(inv_sigmoid(self._network.get_layer_neurons(l)))
            dCdaL = nablaNetwork.get_layer_neurons(l)
            dCdw = (z * dCdaL) * aLminus1 #has same shape as weight matrix
            dCdb = (z * dCdaL) #biases
            
            nablaNetwork.get_network_layer(l).set_weights(-dCdw) #negated
            nablaNetwork.get_network_layer(l).set_biases(-dCdb) #negated
    
        return nablaNetwork.get_network_weights_biases()
    
    def addNablas(nabla1, nabla2):
        """Adds together each element of two nablas in their special format"""
        
        if len(nabla1) != len(nabla2):
            return ValueError("Two nabla vectors are not the same size so cannot add")
        
        summedNabla  = []
        for i in range(len(nabla1)):
            (weights1, weights2) = (nabla1[i][0], nabla2[i][0])
            (biases1, biases2) = (nabla1[i][1], nabla2[i][1])
            
            if (weights1.shape != weights2.shape) or (biases1.shape != biases2.shape):
                return ValueError("Two nabla vectors are not the same size so cannot add")
            else:
                summedNabla.append((weights1+weights2, biases1+biases2))
        return summedNabla
    
    def subtractNablas(nabla1, nabla2):
        """Adds together each element of two nablas in their special format"""
        
        if len(nabla1) != len(nabla2):
            return ValueError("Two nabla vectors are not the same size so cannot add")
        
        summedNabla  = []
        for i in range(len(nabla1)):
            (weights1, weights2) = (nabla1[i][0], nabla2[i][0])
            (biases1, biases2) = (nabla1[i][1], nabla2[i][1])
            
            if (weights1.shape != weights2.shape) or (biases1.shape != biases2.shape):
                return ValueError("Two nabla vectors are not the same size so cannot add")
            else:
                summedNabla.append((weights1-weights2, biases1-biases2))
        return summedNabla
    
    
    def divideNabla(nabla, scalar):
        """Divides  each element of a nabla by a particular scalar and returns it"""
        
        dividedNabla  = []
        for i in range(len(nabla)):
            (weights, biases) = (nabla[i][0], nabla[i][1])
            dividedNabla.append((weights / scalar, biases / scalar))
        return dividedNabla
    
    def get_zero_nabla(network: Network):
        """Create a nabla vector for a particular network with all entries zero."""
        weightsbiases = network.get_network_weights_biases()
        return NetworkTrainer.subtractNablas(weightsbiases, weightsbiases)
        
        
        
        
        
                
        

#Test

test_network = Network([784,16,16,10])
test_network.load_network("overnight3")
alldata = collect_data('../24CookieTrainingData/sample/dataset', [0,1,2,3,4,5,6,7,8,9])
print("DATA COLLECTED")

trainer = NetworkTrainer(test_network, alldata)
test_network = trainer.train_network()
print("DONE!")
test_network.save_network("overnight3")


