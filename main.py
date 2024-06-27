from Helpers import *
from ImageProcessing import *
from Structure import *

test_network = Network([784,16,16,10])
data = get_image_data('../24CookieTrainingData/archive/dataset/0/0.png')
test_network.execute_network(data)
print("CHICKEN")
print(test_network.get_network_weights_biases())