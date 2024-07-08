from Helpers import *
from ImageProcessing import *
from Structure import *
from testing import *
from time import sleep
from playing import *

import tkinter as tk
from PIL import Image, ImageDraw

np.set_printoptions(precision=2, suppress=True)
np.set_printoptions(linewidth=150, precision=2, suppress=True)

test_network = Network([784,16,16,10])
sampledata = collect_data('../24CookieTrainingData/sample/dataset', [0,1,2,3,4,5,6,7,8,9])
test_network.load_network("allfromovernight")
print(test_accuracy(test_network, sampledata))
for i in range(10):
    image = get_image_data(f"../24CookieTrainingData/archive/dataset/{i}/80.png")

realimage = get_image_data("../24CookieTrainingData/validation/big5.png")

print(test_network.execute_network(realimage))

main(test_network)



"""
for label in sampledata:
    for i in range(1050, len(sampledata[label])):
        vector = sampledata[label][i]
        print("================")
        print(f"Should be {label}.")
        print(test_network.execute_network(vector))
        sleep(0.5)
"""
        