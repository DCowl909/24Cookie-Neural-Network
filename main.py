from Helpers import *
from ImageProcessing import *
from Structure import *
from testing import *
from time import sleep

np.set_printoptions(precision=2, suppress=True)

test_network = Network([784,16,16,10])
sampledata = collect_data('../24CookieTrainingData/archive/dataset', [0,1,2,3,4,5,6,7,8,9])
test_network.load_network("overnight3")
print(test_accuracy(test_network, sampledata))



#Testing a bunch of random labels
while True:
    for label in range(10):
        for j in range(1000,1015):
            print(f"SHOULD BE {label}.")
            data = get_image_data(f'../24CookieTrainingData/archive/dataset/{label}/{j}.png')
            print(test_network.execute_network(data))
            sleep(1)
            
            