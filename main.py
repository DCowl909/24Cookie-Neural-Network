from Helpers import *
from ImageProcessing import *
from Structure import *
from playing import *

import tkinter as tk

def test_accuracy(network: Network, data: dict[any, list[np.ndarray]]) -> float:
    """Tests the overall accuracy of a network by determining what percentage of a data set it labels
    correctly."""
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
                
            

np.set_printoptions(linewidth=150, precision=2, suppress=True)

test_network = Network([784,16,16,10])
newdata = collect_data('../24CookieTrainingData/arrays/alldata', [0,1,2,3,4,5,6,7,8,9])
test_network.load_network("newdata1")
print(test_accuracy(test_network, newdata))

root = tk.Tk()
app = DrawApp(root, test_network)
root.mainloop()


