import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np


def invert(arr: np.array):
    full = 255 * np.ones(arr.shape)
    return full - arr

class DrawApp:
    def __init__(self, root, network):
        self.root = root
        self.root.title("Draw a Digit")
        self.canvas = tk.Canvas(self.root, width=840, height=840, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.label = tk.Label(self.root, text="Enter the digit you drew:")
        self.label.pack()
        self.entry = tk.Entry(self.root)
        self.entry.pack()
        
        self.button = tk.Button(self.root, text="Save", command=self.save)
        self.button.pack()
        
        self.image = Image.new("L", (84, 84), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        self.finalarray = None
        self.network = network

    def paint(self, event):
        # Draw the main black circle
        x1, y1 = (event.x - 15), (event.y - 15)
        x2, y2 = (event.x + 15), (event.y + 15)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)
        self.draw.ellipse([x1//10, y1//10, x2//10, y2//10], fill=0, width = 20)

    def save(self):
        # Resize and blur the image
        small_image = self.image.resize((28, 28), Image.LANCZOS)
        
        np_array = np.array(small_image)
        
        np_array = invert(np_array)
        print(np_array)
        np_array = np_array.flatten().reshape(784, 1)
        np_array = self.scale(np_array)
        self.finalarray = np_array
        
        self.save_array()
        self.clear_canvas()
        
    def save_array(self):
        
        np_array = self.finalarray 
        digit = self.entry.get()
        if digit == "":
            print("Did not enter a label.")
            self.test_drawing()
        else:
            npz_path = f"../24CookieTrainingData/arrays/{digit}/D.npz"
            try:
                data = np.load(npz_path)
                existing_arrays = [data[f'array_{i}'] for i in range(len(data.files))]
                combined_arrays = existing_arrays + [np_array]
            except FileNotFoundError:
                combined_arrays = [np_array]
            except KeyError:
                combined_arrays = [np_array]
        
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (84, 84), 255)
        self.draw = ImageDraw.Draw(self.image)
    
    def get_array(self):
        return self.finalarray
    
    def test_drawing(self):
        test_network = self.network
        print(test_network.execute_network(self.finalarray))
        print(f"SO ITS A {test_network.evaluate(self.finalarray)}")

    def scale(self, arr):
        """Scales the input array to the range [0, 1]"""
        return arr / 255.0
