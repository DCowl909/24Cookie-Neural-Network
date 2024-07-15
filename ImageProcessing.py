from PIL import Image
import glob
import numpy as np

def scale(arr: np.ndarray) -> np.ndarray:
    return np.divide(arr, 255)

def invert(arr: np.matrix):
    full = 255 * np.ones(arr.shape)
    return full - arr
        
def get_image_data(image_path: str) -> np.ndarray:
    """Returns an np array of the alpha channels, or average RGB values for each pixel 
       of a 28x28 image. Used as input data for the 24cookie network.
    """
    image = Image.open(image_path)
    if image.size != (28, 28):
        raise ValueError("Image has wrong dimensions")
    
    np_array = np.array(image)
    
    # Check if the image has an alpha channel
    if np_array.shape[2] == 4:
        # Extract the RGB channels and ignore the alpha channel
        np_array = np_array[:, :, 3]
    elif np_array.shape[2] == 3:
        # If the image doesn't have an alpha channel, use the RGB channels as is and invert
        np_array = np_array
        np_array = np.mean(np_array, axis=2)
        np_array = invert(np_array)
    else:
        raise ValueError("Image format not recognized")
    # Flatten the array to a column vector
    #print(np_array)
    flat_array = np_array.flatten().reshape(784, 1)
    
    image.close()
    return scale(flat_array)


def collect_files(image_folder_path: str, type: str) -> list[str]:
    """Obtains a list of all the directories of files inside a folder path of a specified file type.
        For example, entering type as "png" will return all the pngs. 
    """
    image_files = []
    
    image_files.extend(glob.glob(f"{image_folder_path}/*.{type}"))
    return image_files    


def process_images(imageDirectory: str, arrayDirectory: str, labels:list) -> None:
        """Processes a large amount of data by turning each png into an array and saving them in
        a directory. The array directory will have an identical structure but instead of being filled
        with pngs will be filled with a npz file containing all the arrays of that label.

        Args:
            imageDirectory (str): The relative directory to obtain the training data. This method 
                expects the data to be organised such that each piece of data with the same label
                is in it's own folder, where the folder is named the label.
                
            arrayDirectory (array): The relatvie directory to save the processed training data.
            
            labels (list[Any]): the list of labels for the data. These should match the names of
                the folders in the directory. Order of these does not need to match order of folders.
                
        Returns: 
            A dictionary with each key being the label and each value being a list of the 
            data vectors associated with that label.
        """
        #Store and sorts all the data vectors to their label
        
        for label in labels:
            imagePaths = collect_files(f"{imageDirectory}/{label}", "png")
            processedData = [get_image_data(path) for path in imagePaths]
            dataMatrix = np.hstack(processedData)
            np.save(f"{arrayDirectory}/{label}/originaldata.npy", dataMatrix)
            print(f"Processed and saved data for label {label}.")

import pandas as pd

def process_csv(csvDirectory: str, arrayDirectory: str):
    """Processes a csv file of hand drawn numerical data into a data matrix format this network
    can read. Assumes the csv has each data vector in a row seperated by commas, with the first 
    entry in a row being the label the rest of the row is supposed to be."""
    df = pd.read_csv(csvDirectory)
    np_matrix = df.to_numpy()
    np_matrix = np_matrix.T
    
    labels = np_matrix[0]
    data = np_matrix[1:]
    
    label_matrices = {i: [] for i in range(10)}
    for i in range(data.shape[1]):
        label = int(labels[i])
        label_matrices[label].append(data[:, i])
    
    # Convert lists to NumPy matrices
    for label in label_matrices:
        label_matrices[label] = np.array(label_matrices[label]).T
        
    for label in label_matrices:
        np.save(f"{arrayDirectory}/{label}/secondarydata.npy", label_matrices[label] /255)




def collect_data(arrayDirectory: str, labels: list) -> dict[int, list[np.array]]:
    """Processes stored, preprocessed data into a dictionary the learning algorithm can use."""
    sortedData = {}
    for label in labels:
        sortedData[label] = []
        for file_directory in collect_files(f"{arrayDirectory}/{label}", "npy"):
            dataMatrix = np.load(file_directory, allow_pickle=True)
            sortedData[label] += [dataMatrix[:, i:i+1] for i in range(dataMatrix.shape[1] - 1)] #extracts every column of the matrix
        print(f"Loaded data for label {label}. Amount of data = {(len(sortedData[label]))}")
                
    return sortedData
