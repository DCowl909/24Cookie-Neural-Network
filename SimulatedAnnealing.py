import numpy as np
from Structure import *
from Learning import cost, totalCost, expected_final_layer

import pylab
import math
import random

def partialCost(network: Network, data: dict[any, list[np.ndarray]]):
    totalCost = 0
    i = 0
    for label in data:
        expected = expected_final_layer(label)
        for vector in data[label]:
            if i%100 == 0:
                actual = network.execute_network(vector)
                totalCost += cost(expected, actual)
            i+= 1
    return totalCost
        
    
#the solution for this simulated annealing is the network weights and biases
def RunSA(Solution,Cost,ChooseNeigh,MoveToNeigh,T,N,alpha, data) -> Network:
    E = Cost(Solution, data)
    Best = E
    CostArr = [E]
    BestArr = [Best]
    BestSol = Solution  # since Solution will be a array (weights and biases)
    for i in range(N):
        print(f"{100*i/N}%")
        delta,neighbour = ChooseNeigh(Solution, data)
        if delta < 0 or math.exp(-delta/T) > random.random():
            MoveToNeigh(Solution,neighbour)
            E += delta
            if E < Best:
                Best = E
                BestSol = Solution
        CostArr.append(E)
        BestArr.append(Best)
        T *= alpha
    print (Best, T)
    pylab.plot(range(N+1),CostArr)
    pylab.plot(range(N+1),BestArr)
    pylab.show()
    return BestSol

def NeighbourNetwork(network: Network, data: dict[any, list[np.ndarray]]):
    #choose a neighbour solution by picking a random weight and bias and changing it by 0.1
    currCost = totalCost(network, data)
    
      # Get the current weights and biases
    weightsbiases = network.get_network_weights_biases()
    
    # Store the original weights and biases to restore later
    original_weightsbiases = [(w.copy(), b.copy()) for (w,b) in weightsbiases]
    
    for _ in range(5):
        l = np.random.randint(0,len(weightsbiases))
        n = np.random.randint(0,2)
        rows, cols = weightsbiases[l][n].shape
        random_row, random_col = np.random.randint(0, rows), np.random.randint(0, cols) 
        
        sign = np.random.choice([1, -1])
        weightsbiases[l][n][random_row, random_col] += sign * 0.01 #edit the chosen w/b
    
    network.set_network_weights_biases(weightsbiases)
    
    delta = totalCost(network, data) - currCost
    
    network.set_network_weights_biases(original_weightsbiases)
    
    return delta, weightsbiases

def MoveNetwork(network: Network, neighbour: dict[any, list[np.ndarray]]):
    
    network.set_network_weights_biases(neighbour)
    

#startingNetwork = Network([784, 16, 16, 10])
#startingNetwork.load_network("newdata1")
#alldata = collect_data('../24CookieTrainingData/arrays/alldata', [0,1,2,3,4,5,6,7,8,9])
#startingNetwork = RunSA(startingNetwork, totalCost, NeighbourNetwork , MoveNetwork, 10, 800, .999, alldata )