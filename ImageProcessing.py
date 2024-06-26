from PIL import Image
import numpy as np

def scale(arr: int) -> int:
    return np.divide(arr, 255)

def get_image_data(image_path: str) -> np.array:
    """Returns an np array of the alpha values for each pixel 
        of a 28x28 image. Used as input data for the 24cookie network.
    """
    image = Image.open(image_path)
    if image.size != (28, 28):
        raise ValueError("Image has wrong dimensions")
    
    np_array = np.array(image)
    if np_array.shape[2] != 4:
        raise ValueError("Image does not have an alpha channel")
    
    # Extract the alpha channel
    alpha_np_array = np_array[:, :, 3]
    #flatten the array to a column vector
    alpha_flat_array = alpha_np_array.flatten().reshape(784, 1)
    
    image.close()
    return scale(alpha_flat_array)
