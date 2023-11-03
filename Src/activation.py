import numpy as np


def linear(x):
    return x
def reLu(x):
    return np.maximum(0,x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
    return exp_x / exp_x.sum(axis=0, keepdims=True)
def tanh(x):
    return np.tanh(x)

    

def derived_linear(x):
    return 1

def derived_RELU(x):
    return 0 if x < 0 else 1

def derived_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def derived_softmax(x, target):
    return x if 1 != target else -(1-x)
