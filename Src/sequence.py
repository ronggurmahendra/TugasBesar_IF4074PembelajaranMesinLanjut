import numpy as np
import math
from utils import *
import json

# bias convolution - OKAY
# update dense bias
# update convolution bias
# handling batch
# update weight based on batch
# update bias based on batch

class Sequence: 

    def __init__(self, input_shape_ = None):
        self.layers = []
        self.is_compiled = False
        self.input_shape = input_shape_

        if len(self.input_shape) == 2:
            self.input_shape = (1, self.input_shape[0], self.input_shape[1])
        
        self.feeding_shape = self.input_shape
    
    def add(self, layer_):
        self.layers.append(layer_)

    def predict(self, input_):
        # add a dimension to a single channel input
        if self.is_compiled == False:
            raise ValueError("Model is not compiled yet")
        if len(input_.shape) == 2:
            input_ = np.expand_dims(input_, axis=0)
        if len(input_.shape) == 3:
            input_ = np.expand_dims(input_, axis=0)
        predictions = []
        for d in input_:
            for layer in self.layers:
                output = layer.feedForward(d)
                d = output
            if output.shape[0] == 1:
                output = np.squeeze(output)
            predictions.append(output)
        return np.array(predictions)
    
    def fit(self, input_, Y, epochs, loss="log_loss", learning_rate=0.01, batch_size=1):
        if self.is_compiled == False:
            raise ValueError("Model is not compiled yet")
        
        if len(input_.shape) == 2:
            input_ = np.expand_dims(input_, axis=0)
        if len(input_.shape) == 3:
            input_ = np.expand_dims(input_, axis=0)
        
        int_dict = {i: [] for i in range(len(self.layers))}
        
        data = input_
        for e in range(epochs):
            output = []
            input_ = data
            # feed forwarding
            for i, d in enumerate(input_):
                for j in range(len(self.layers)):
                    layer = self.layers[j]
                    output = layer.feedForward(d)
                    int_dict[j] = output
                    d = output
                for j in range(len(self.layers)-1, -1, -1):
                    layer = self.layers[j]
                    if (j == len(self.layers)-1):
                        if loss == "log_loss":
                            output_class = Y[i]
                            layer.dOutput = np.array([p if i != output_class else p - 1 for i, p in enumerate(layer.output)])
                        elif loss == "mse":
                            layer.dOutput = np.array([p - Y[i] for p in layer.output]) * DetectorLayer.sigmoid_derivative(layer.output)
                        dE_dW = np.array(multiply_arrays(layer.dOutput, layer.prev_layer.output))
                        # update weight
                        layer.weights -= dE_dW.T * learning_rate
                        layer.bias -= layer.dOutput * learning_rate
                        layer.dWn = dE_dW
                    else:
                        dE_dW = layer.backprop()
            output = self.predict(data)
            print("\n")
            print("Epoch ", e + 1)
            accuracy = 0
            if loss == "log_loss":
                for i, o in enumerate(output):
                    if np.argmax(o) == Y[i]:
                        accuracy += 1
                accuracy /= len(output)
                print("Accuracy: ", accuracy)
            elif loss == "mse":
                for i, o in enumerate(output):
                    if o > 0.5:
                        o = 1
                    else:
                        o = 0
                    if o == Y[i]:
                        accuracy += 1
                accuracy /= len(output)
                print("Accuracy: ", accuracy)

    def load_model(self, filename):
        
        loaded_layers = []

        with open(filename, "r") as json_file:
            serialized_layers = json.load(json_file)
        
        for layer_data in serialized_layers:
            if layer_data["type"] == "ConvolutionLayer":
               
                filter_num_ = layer_data["params"]["filter_num"]
                filter_size_ = layer_data["params"]["filter_size"]
                filter_ = layer_data["params"]["filter"]
                bias_ = layer_data["params"]["bias"]
                padding_ = layer_data["params"]["padding"]
                stride_ = layer_data["params"]["stride"]
                input_shape_ = layer_data["params"]["input_shape"]

                layer = ConvolutionLayer(filter_num_, filter_size_, padding_, stride_, input_shape_)
                layer.filter = filter_
                layer.filter = bias_
            elif layer_data["type"] == "DetectorLayer":
                layer = DetectorLayer()
            elif layer_data["type"] == "PoolingLayer":
                kernel_size_ = layer_data["params"]["kernel_size"]
                stride_ = layer_data["params"]["stride"]
                mode_ = layer_data["params"]["mode"]
                
                layer = PoolingLayer(kernel_size_, mode_, stride_ = stride_)
            elif layer_data["type"] == "FlattenLayer":
                layer = FlattenLayer()
            elif layer_data["type"] == "DenseLayer":

                units_ = layer_data["params"]["units"]
                weights_ = layer_data["params"]["weights"]
                bias_ = layer_data["params"]["bias"]
                activation_ = layer_data["params"]["activation"]
                output_ = layer_data["params"]["output"]
                input_  = layer_data["params"]["input"]

                layer = DenseLayer(units_, activation_)
                layer.bias = bias_
                layer.weights = layer.weights_
                layer.output = output_
                layer.input = input_
            else:
                print("Layer not detected, something went wrong")
            loaded_layers.append(layer)
        self.layers = loaded_layers

        return 1

    def save_model(self, filename):
        serialized_layers = []
        for layer in self.layers:
            if isinstance(layer, ConvolutionLayer):
                layer_data = {
                    "type" : "ConvolutionLayer",
                    "params" : {
                        "filter_num": layer.filter_num,
                        "filter_size": layer.filter_size,
                        "filter": layer.filter.tolist(),
                        "bias": layer.bias,
                        "padding": layer.padding,
                        "stride": layer.stride,
                        "input_shape": layer.input_shape,
                    }
                }
            elif isinstance(layer,DetectorLayer):
                layer_data = {
                    "type" : "DetectorLayer",
                    "params" : {

                    }
                }
            elif isinstance(layer,PoolingLayer):
                layer_data = {
                    "type" : "PoolingLayer",
                    "params" : {
                        "kernel_size" : layer.kernel_size,
                        "stride" : layer.stride,
                        "mode" : layer.mode,
                    }
                }
            elif isinstance(layer,FlattenLayer):
                layer_data = {
                    "type" : "FlattenLayer",
                    "params" : {}
                }
            elif isinstance(layer,DenseLayer):
                layer_data = {
                    "type" : "DenseLayer",
                    "params" : {
                        "units" : layer.units ,
                        "weights" : layer.weights ,
                        "bias" : layer.bias ,
                        "activation" : layer.activation,
                        "output" : layer.output,
                        "input" : layer.input,
                    }
                }
            else:
                print("Layer not detected, something went wrong")
                return
            serialized_layers.append(layer_data)

        with open(filename, "w") as json_file:
            json.dump(serialized_layers, json_file)

        return 1

    def compile(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.compile(self, next_layer=self.layers[i+1])
            elif (i == len(self.layers)-1):
                layer.compile(self.layers[i-1])
            else:
                layer.compile(self.layers[i-1], next_layer=self.layers[i+1])
        self.is_compiled = True


class ConvolutionLayer:
    filter_num = -1
    filter_size = [-1,-1]
    filter = [[]]
    bias = 1
    padding = -1
    stride = -1
    input_shape = [-1,-1,-1]
    output = None
    next_layer = None
    input_matrix = None
    dWn = None

    def __init__(self, filter_num_, filter_size_, padding_, stride_, input_shape_ = [-1,-1]):
        self.filter_num = filter_num_
        self.filter_size = filter_size_
        self.padding = padding_
        self.stride = stride_
        self.input_shape = input_shape_
        self.filter = np.random.rand(self.filter_num,self.filter_size[0], self.filter_size[1])
        self.bias = np.zeros(self.filter_num)

        

    def feedForward(self, input_):
        self.input_matrix = input_
        input_height_padded = input_.shape[1] + 2 * self.padding
        input_width_padded = input_.shape[2] + 2 * self.padding
        
        input_padded = np.zeros((input_.shape[0], input_height_padded, input_width_padded))
        input_padded[ : , 
               self.padding: self.padding + input_.shape[1],
               self.padding: self.padding + input_.shape[2]] = input_
        
        
        channel, height, width = input_padded.shape

        conv_height = (height - self.filter_size[0] + 1) // self.stride[0]
        conv_width = (width - self.filter_size[1] + 1) // self.stride[1]
    
        output = np.zeros((self.filter_num, conv_height, conv_width)) # output feature map
        
        for f in range (self.filter_num):
            for i in range(conv_height):
                for j in range(conv_width):
                    conv_region = input_padded[:,i*self.stride[0]:i*self.stride[0] + self.filter_size[0], j*self.stride[1]:j*self.stride[1] + self.filter_size[1]]
                    output[f, i, j] += np.sum(conv_region * self.filter[f]) + self.bias[f]
        self.output = output
        return output
    
    def compile(self, prev_layer_, next_layer=None):
        if next_layer != None:
            self.next_layer = next_layer
        self.input_shape = prev_layer_.feeding_shape
        input_height_padded = prev_layer_.feeding_shape[1] + 2 * self.padding
        input_width_padded = prev_layer_.feeding_shape[2] + 2 * self.padding
        if len(prev_layer_.feeding_shape) != 3:
            raise ValueError("Invalid input shape for Convolution Layer")
        self.feeding_shape = (self.filter_num, (input_height_padded - self.filter_size[0] + 1) // self.stride[0], (input_width_padded - self.filter_size[1] + 1) // self.stride[1])
    
    def backprop(self, learning_rate=0.01):
        input_ = self.input_matrix
        input_height_padded = input_.shape[1] + 2 * self.padding
        input_width_padded = input_.shape[2] + 2 * self.padding
        
        input_padded = np.zeros((input_.shape[0], input_height_padded, input_width_padded))
        input_padded[ : , 
               self.padding: self.padding + input_.shape[1],
               self.padding: self.padding + input_.shape[2]] = input_
        
        
        channel, height, width = input_padded.shape

        conv_height = (height - self.next_layer.dWn.shape[1] + 1) // self.stride[0]
        conv_width = (width - self.next_layer.dWn.shape[2] + 1) // self.stride[1]
    
        output = np.zeros((self.filter_num, conv_height, conv_width)) # output feature map
        
        for f in range (self.filter_num):
            for i in range(conv_height):
                for j in range(conv_width):
                    conv_region = input_padded[:,i*self.stride[0]:i*self.stride[0] + self.next_layer.dWn.shape[1], j*self.stride[1]:j*self.stride[1] + self.next_layer.dWn.shape[2]]
                    output[f, i, j] += np.sum(conv_region * self.next_layer.dWn[f])
        self.dWn = output
        dE_dB = np.sum(self.next_layer.dWn, axis=(1, 2))
        self.filter = np.array([f - learning_rate * df for f, df in zip(self.filter, self.dWn)])
        self.bias = np.array([f - learning_rate * df for f, df in zip(self.bias, dE_dB)])
        return self.dWn

class DetectorLayer:
    output = None
    next_layer = None
    dWn = None
    def feedForward(self, input_):
        channel, height, width = input_.shape
        output = np.zeros((channel, height, width)) # output feature map
        for c in range (channel):
            for i in range(height):
                for j in range(width):
                    output[c,i,j] = DetectorLayer.reLu(input_[c,i,j])
        self.output = output
        return output
    
    def reLu(x):
        return np.maximum(0,x)
    def sigmoid(x):
        if x < 0:
            return np.exp(x) / (1 + np.exp(x))
        else:
            return 1/(1+np.exp(-x))
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
        return exp_x / exp_x.sum(axis=0, keepdims=True)
    def compile(self, prev_layer_, next_layer=None):
        if next_layer != None:
            self.next_layer = next_layer
        self.feeding_shape = prev_layer_.feeding_shape
    def backprop(self):
        self.dWn = self.next_layer.dWn
        return self.dWn
    def derived_softmax(x, target):
        return x if 1 != target else -(1-x)
    def reLu_derivative(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def sigmoid_derivative(x):
        return DetectorLayer.sigmoid(x) * (1 - DetectorLayer.sigmoid(x))
    def compile(self, prev_layer_, next_layer=None):
        if next_layer != None:
            self.next_layer = next_layer
        self.feeding_shape = prev_layer_.feeding_shape
class PoolingLayer:
    kernel_size = [-1,-1]
    stride = 0 
    mode = "" # either max / average
    output = None
    dOutput = None
    prev_output = None
    max_pool_index = []
    dWn = None

    def __init__(self, filter_size_, mode_, stride_=None):
        self.filter_size = filter_size_
        # definition of stride is define as
        # https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
        if stride_ is None:
            self.stride = filter_size_
        self.mode = mode_

    def feedForward(self, input_):
        self.prev_output = input_
        self.max_pool_index = []
        channel, input_height, input_width = input_.shape
        output_height = input_height // self.stride[0]
        output_width = input_width // self.stride[1]
        
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Invalid input or filter size")

        output = np.zeros((channel, output_height, output_width))

        for ch in range(channel):
            for i in range(output_height):
                for j in range(output_width):
                    output[ch,i,j], row, col = self.pooling(input_[
                        ch,
                        i*self.stride[0]:(i+1)*self.filter_size[0],
                        j*self.stride[1]:(j+1)*self.filter_size[1]
                        ])
                    self.max_pool_index.append((row, col))
        self.output = output
        return output
    
    def compile(self, prev_layer_, next_layer=None):
        if next_layer != None:
            self.next_layer = next_layer
        channel, input_height, input_width = prev_layer_.feeding_shape
        self.feeding_shape = (channel, input_height // self.stride[0], input_width // self.stride[1])
    
    def pooling(self, input_):
        if self.mode == "max":
            i, j = np.unravel_index(np.argmax(input_), input_.shape)
            return np.max(input_), i, j
        elif self.mode == "average":
            return np.average(input_), 0, 0
        
    def backprop(self):
        self.dOutput = multiply_and_sum(self.next_layer.dOutput, self.next_layer.weights)
        if self.mode == 'max':
            input_ = self.prev_output
            channel, input_height, input_width = input_.shape
            output_height = input_height // self.stride[0]
            output_width = input_width // self.stride[1]

            output = np.zeros(input_.shape)

            for ch in range(channel):
                
                output[ch, self.max_pool_index[ch][0], self.max_pool_index[ch][1]] = self.dOutput[ch]
            self.dWn = output
            return self.dWn

          


        #TODO: Handle for Average pooling, tapi kayaknya ga diajarin?
        return [0,0,0,0]

class FlattenLayer:
    output = None
    next_layer = None
    dOutput = None
    weights = None
    dWn = None
    def feedForward(self, input_):
        self.output = np.reshape(input_, (-1))
        return np.reshape(input_, (-1))
    
    def compile(self, prev_layer_, next_layer=None):
        if next_layer != None:
            self.next_layer = next_layer
        total = 1
        for x in prev_layer_.feeding_shape:
            total *= x
        self.feeding_shape = (total,)
    
    def backprop(self):
        self.dOutput = self.next_layer.dOutput
        self.weights = self.next_layer.weights
        self.dWn = self.next_layer.dWn
        return self.next_layer.dWn

class DenseLayer:
    output = None
    prev_layer = None
    next_layer = None
    weights = None
    output_no_activation = None
    name = "dense"
    dOutput = None
    dWn = None
    
    def __init__(self, units_, activation_, weights_=[]):
        self.units = units_
        self.weights = weights_
        self.activation = activation_
        self.bias = []

    def feedForward(self, input_):
        # for loop through input
        return np.array(self.ff(input_))

    def ff(self, entity):
        if self.activation == "relu":
            self.output = np.array(list(map(lambda x: DetectorLayer.reLu(x),np.dot(entity, self.weights))))
            return np.array(list(map(lambda x: DetectorLayer.reLu(x),np.dot(entity, self.weights) + self.bias)))
        elif self.activation == "sigmoid":
            self.output = np.array(list(map(lambda x: DetectorLayer.sigmoid(x),np.dot(entity, self.weights))))
            return np.array(list(map(lambda x: DetectorLayer.sigmoid(x),np.dot(entity, self.weights) + self.bias)))
        elif self.activation == "softmax":
            self.output = DetectorLayer.softmax(np.dot(entity, self.weights))
            return DetectorLayer.softmax(np.dot(entity, self.weights) + self.bias) 

    def dO_dW(self):
        return self.prev_layer.output
    
    def dE_dO(self):
        if self.activation == 'relu':
            self.dOutput = self.dReLu(multiply_and_sum(self.next_layer.dOutput, self.next_layer.weights))
            return self.dOutput
    
    def dReLu(self, input_arr):
        for i in range(len(input_arr)):
            if self.output[i] == 0:
                input_arr[i] = 0
        return input_arr

    def backprop(self, learning_rate=0.01):
        self.dWn = multiply_arrays(self.dE_dO(), self.dO_dW())
        self.weights -= self.dWn.T * learning_rate
        self.bias -= self.dE_dO() * learning_rate
        return self.dWn

    def compile(self, prev_layer_, next_layer=None):
        if next_layer != None:
            self.next_layer = next_layer
        self.prev_layer = prev_layer_
        if len(prev_layer_.feeding_shape) != 1:
            raise ValueError("Invalid input shape for Dense Layer")
        self.weights = np.random.rand(prev_layer_.feeding_shape[0], self.units)
        self.bias = np.random.rand(self.units)
        self.feeding_shape = (self.units,)