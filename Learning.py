#33% faster using arrays

import random
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

def totalCost(network: Network, data: dict[any, list[np.ndarray]]):
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

def process_nablaC_vector(nablaCVector: np.ndarray, networkStructure: list[int]) -> list[tuple[np.matrix, np.ndarray]]:
    """Converts a vector representing the cost gradient, in which each element is the partial derivative of a weight or bias with respect to cost, 
    into a format that a network can read."""
    cutFrom = 0
    result = []

    for i in range(1, len(networkStructure)):
        weightDimensions = (networkStructure[i], networkStructure[i-1])
        biasDimensions = (networkStructure[i], 1)
        
        # Extract weights
        weightSize = weightDimensions[0] * weightDimensions[1]
        flattenedWeights = nablaCVector[cutFrom : cutFrom + weightSize]
        weights = flattenedWeights.reshape(weightDimensions)
        cutFrom += weightSize
        
        # Extract biases
        biasSize = biasDimensions[0]
        biases = nablaCVector[cutFrom : cutFrom + biasSize]
        cutFrom += biasSize
        
        result.append((weights, biases))
    
    return result
        

    

class NetworkTrainer():
    
    def __init__(self, network: Network, data: dict[any, list[np.ndarray]]):
        self._network = network
        self._networkDepth = network.get_depth()
        self._data = data
        self._dataSize = dataSize(data)
        
    def train_network(self, startingFactor: int) -> Network:
        iterate = 160
         
        factor = startingFactor
        previousCost = totalCost(self._network, self._data)
        for i in range(iterate):
            nablaCost = self.total_stochastic_nabla_cost() #big column vector with length of network complexitiy
            processedNablaCost = process_nablaC_vector(-factor * nablaCost, self._network._structure)
            self._network.change_network_weights_biases(processedNablaCost)

            
            if (i%20 == 0):
                currentCost = totalCost(self._network, self._data)
                print(f"Current total cost is {totalCost(self._network, self._data)}")
                print(f"Progress: {100*i/iterate}%")
                
                if (previousCost - currentCost) < 0:
                    factor = factor * 0.2
                else:
                    factor = factor * 1.2
                print(f"Factor: {factor}")
                previousCost = currentCost
            
        return self._network
            
        
    def total_nabla_cost(self) -> np.ndarray:
        
        """Uses the backpropogation algorithm to determine the current nabla vector of the cost function. 
            Only works with the final layer having 10 neurons. Returned vector has flattened weights of layer 1, then
            layer 2 and so on."""
            
        summedNablaC = np.zeros(self._network._complexity).reshape(-1,1)
        
        for label in self._data:
            for vector in self._data[label]:
                dataVectorNablaC = self.local_nabla_cost(vector, label)
                summedNablaC = summedNablaC + dataVectorNablaC 
                
        return summedNablaC / self._dataSize
    
    def total_stochastic_nabla_cost(self) -> np.ndarray:
        
        """Uses the backpropogation algorithm to determine the current nabla vector of the cost function. 
            Only works with the final layer having 10 neurons. Returned vector has flattened weights of layer 1, then
            layer 2 and so on."""
            
        summedNablaC = np.zeros(self._network._complexity).reshape(-1,1)
        batchSize = 0
        
        for label in self._data:
            
            random_integers = [random.randint(0, 1000) for _ in range(200)]
            
            for integer in random_integers:
                vector = self._data[label][integer]
                dataVectorNablaC = self.local_nabla_cost(vector, label)
                summedNablaC = summedNablaC + dataVectorNablaC 
                batchSize += 1
                
        return summedNablaC / batchSize
                
        
    def local_nabla_cost(self, dataVector: np.ndarray, label: int) -> np.ndarray:
        """Uses the backpropogation algorithm to determine the current downhill nabla vector of the cost function 
        with a SPECIFIC DATA VECTOR. This returns -nabla_C."""
        
        #start at the last layer, and calculate all the dCostdNeurons
        #then calculate weights
        #then calculate biases
        
        self._network.execute_network(dataVector)
        n = self._network
        
        nablaC = np.empty(0).reshape(-1,1)
        
        #calculate all of the dCostdNeurons
        dCdNeurons = [None for l in range(self._networkDepth)]
        for l in range(self._networkDepth- 1, 0 , -1):
            if l == self._networkDepth - 1:
                dCdaL = 2 * (n.get_neurons(l) - expected_final_layer(label)) #vector of dCostdNeurons
                
            else:
                dCdaL = np.zeros(n.get_layer_size(l)).reshape(-1, 1)
                #j is nuerons in layer l + 1
                for j in range(n.get_layer_size(l+1)):
                    
                    w = n.get_layer_weights(l+1)[j].reshape(-1,1)
                    daLplus1daL = w * d_sig(n.get_unstandardised_neurons(l+1)[j])
                    dCdaL += dCdNeurons[l+1][j] * daLplus1daL
        
            dCdNeurons[l] = dCdaL
            
        #calculate all of the weights  and biases
        for l in range(self._networkDepth- 1, 0 , -1):
            
            aLminus1 = n.get_neurons(l-1).reshape(-1)
            z = d_sigmoid(n.get_unstandardised_neurons(l))
            dCdaL = dCdNeurons[l]
            dCdw = (z * dCdaL) * aLminus1 #has same shape as weight matrix
            dCdb = (z * dCdaL) #biases vector
            
            flattened_weights = np.ravel(dCdw).reshape(-1,1)
        
            nablaC = np.vstack((flattened_weights, dCdb, nablaC,)) #ravel?

        return nablaC
        
        
        
        
            
        

#Overnight All Test

test_network = Network([784,16,16,10])
test_network.load_network("allfromovernight")
alldata = collect_data('../24CookieTrainingData/sample/dataset', [0,1,2,3,4,5,6,7,8,9])
print(f"DATA COLLECTED. Data Size = {dataSize(alldata)}")

trainer = NetworkTrainer(test_network, alldata)
f = 5

while True:
    test_network.load_network("allfromovernight")

    startingCost = totalCost(test_network, alldata)

    test_network = trainer.train_network(f)
    print("DONE!")

    endingCost = totalCost(test_network, alldata)
    if (endingCost < startingCost):
        test_network.save_network("allfromovernight")
        print("SAVING...")
    else:
        f = f*0.9

