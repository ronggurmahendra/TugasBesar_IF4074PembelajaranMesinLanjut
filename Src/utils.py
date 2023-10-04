from numpy import asarray
from PIL import Image
import os
import numpy as np


def createPath(folder):
    return ROOT + '/' + folder

def readDatasets():
    os.chdir(PANDA_PATH)
    pandas = []
    with os.scandir(PANDA_PATH) as files:
        for file in files:
            if file.name.endswith('.jpeg'):
                pandas.append(file.name)
    return pandas

def createMatrix(dataset, idelta_x):
    image = Image.open(PANDA_PATH+'/'+dataset[idelta_x])
    squared = asarray(image)

    # squared = squaredPadding(data)
    red = separateChannel(squared, 0)
    green = separateChannel(squared, 1)
    blue = separateChannel(squared, 2)
    split_matrix = np.array([red,green,blue])
    return split_matrix

# TODO: still error, for milestone 2 tho (multiple files)
def loadData(dataset):
    data_matrix = []
    for i in range(len(dataset)):
        data_matrix.append(createMatrix(dataset, i))
    return data_matrix 

def separateChannel(color_one,idelta_x):
    height = len(color_one)
    width = len(color_one[0])
    color = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            color[i][j] = color_one[i][j][idelta_x]
    return color
    
def squaredPadding(RGB_Matrix):
    height, width = RGB_Matrix.shape[0], RGB_Matrix.shape[1]
    matrix = np.zeros((height,width,3),dtype=int)

    if (height>width):
        new_height = height
        new_width = height
    else : 
        new_height = width
        new_width = width
    padding_h = int((new_height - height)/2)
    padding_w = int((new_width - width)/2)

    matrix = np.zeros((new_height,new_width,3),dtype=int)
    
    matrix[padding_h:height + padding_h, padding_w:width + padding_w] = RGB_Matrix

    return matrix

def multiply_arrays(arr1, arr2):
    """
    Multiply each element in arr1 with each element in arr2.

    Parameters:
    arr1 (numpy.ndarray): First array.
    arr2 (numpy.ndarray): Second array.

    Returns:
    numpy.ndarray: Resulting 2x10 matrix.
    """
    result = np.zeros((len(arr1), len(arr2)))
    
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            result[i][j] = arr1[i] * arr2[j]

    return result


def multiply_and_sum(arr1, arr2):
    """
    Multiply each row of arr2 by the array arr1 element-wise and sum the products for each row.

    Parameters:
    arr1 (numpy.ndarray): First array.
    arr2 (numpy.ndarray): Second array.

    Returns:
    numpy.ndarray: Resulting array after summing the products for each row.
    """
    # Reshape arr1 to match the dimensions of arr2
    arr1_reshaped = arr1.reshape(1, -1)

    # Multiply each row of arr2 with arr1 element-wise
    multiplied = arr2 * arr1_reshaped

    # Sum the products for each row
    result = np.sum(multiplied, axis=1)
    return result

ROOT = os.path.dirname(os.path.abspath(__file__)) +'../../data'
PANDA_PATH = createPath("milestone_1")


