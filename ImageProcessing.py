from PIL import Image
import glob
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


def collect_images(image_folder_path: str) -> list[str]:
    """Obtains a list of all the directories of png files inside a folder path.
    """
    image_files = []
    
    image_files.extend(glob.glob(f"{image_folder_path}/*.png"))
    return image_files    


def collect_data(directory: str, labels:list) -> dict[any, list[np.ndarray]]:
        """Creates a dictionary of sorted data by assigning each data vecot to it's label.
            Currently only works for pngs

        Args:
            directory (str): The relative directory to obtain the training data. This method 
            expects the data to be organised such that each piece of data with the same label
                is in it's own folder, where the folder is named what the label is.
            labels (list[Any]): the list of labels for the data. These should match the names of
                the folders in the directory. Order of these does not need to match order of folders.
                
        Returns: 
            A dictionary with each key being the label and each value being a list of the 
            data vectors associated with that label.
        """
        #Store and sorts all the data vectors to their label
        sortedData = {}
        for label in labels:
            imagePaths = collect_images(f"{directory}/{label}")
            sortedData[label] = [get_image_data(path) for path in imagePaths]
        return sortedData
    