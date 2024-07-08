from ImageProcessing import *
from Structure import *

def test_accuracy(network: Network, data) -> float:
    total = 0
    correct = 0
    for label in data:
        for dataVector in data[label]:
            total += 1
            if network.evaluate(dataVector) == label:
                correct += 1
            #else:
                #print(f"SHOULD BE {label}")
                #print(network.execute_network(dataVector))
        print(f"Current accuracy = {100* correct/total}%")
        
    return correct/total
                
            