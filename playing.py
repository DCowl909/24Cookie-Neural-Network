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
        self.button = tk.Button(self.root, text="Save", command=self.save)
        self.button.pack()
        self.image = Image.new("L", (84, 84), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        self.finalarray = 0
        self._network = network

    def paint(self, event):
        # Draw the main black circle
        x1, y1 = (event.x - 15), (event.y - 15)
        x2, y2 = (event.x + 15), (event.y + 15)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=2)
        self.draw.ellipse([x1//10, y1//10, x2//10, y2//10], fill=0)

        # Draw surrounding gray circles for gradient effect
        for i, shade in enumerate([150, 200, 230]):
            x1, y1 = (event.x - 15 - i), (event.y - 15 - i)
            x2, y2 = (event.x + 15 + i), (event.y + 15 + i)
            self.canvas.create_oval(x1, y1, x2, y2, outline='', fill=f'#{shade:02x}{shade:02x}{shade:02x}')
            self.draw.ellipse([x1//10, y1//10, x2//10, y2//10], fill=255 - shade)

    def save(self):
        # Resize and blur the image
        small_image = self.image.resize((28, 28), Image.LANCZOS)
        np_array = np.array(small_image)
        np_array = invert(np_array)
        print(np_array)
        np_array = np_array.flatten().reshape(784, 1)
        np_array = self.scale(np_array)
        self.finalarray = np_array
        self.root.destroy()
        
        print(self._network.execute_network(np_array))
        print(f"ITS A {self._network.evaluate(np_array)}")
    
    def get_array(self):
        return self.finalarray

    def scale(self, arr):
        """Scales the input array to the range [0, 1]"""
        return arr / 255.0

def main(network):
    root = tk.Tk()
    app = DrawApp(root, network)
    root.mainloop()

if __name__ == "__main__":
    main()
