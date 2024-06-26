from Helpers import *
from ImageProcessing import *
from Structure import *

test_network = Network([4,3,2])
data = [0.1, 0.7, 0.7, 0.3]
test_network.execute_network(data)