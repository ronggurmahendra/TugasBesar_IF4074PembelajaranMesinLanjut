from activation_function import sigmoid , softmax
from Activation import Activation
import numpy as np


def linear(x):
    return x

def RELU(x):
    return max(0,x)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(x):

    maxEl = max(x)
    sum_value = 0
    for value in x:
        sum_value += np.exp(value - maxEl)
    arr_result = []

    for value in x:
        arr_result.append(np.exp(value - maxEl)/sum_value)
        if math.isnan(arr_result[-1]):
            print("Is nan", x)
    return arr_result
    

def derived_linear(x):
    return 1

def derived_RELU(x):
    return 0 if x < 0 else 1

def derived_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def derived_softmax(x, target):
    return x if 1 != target else -(1-x)
