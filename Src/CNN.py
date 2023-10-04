import numpy as np

from utils import *
verbose = True
class CNN: 

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

        output = []  
        for layer in self.layers:
            output = layer.feedForward(input_)
            input_ = output
        return output
    
    def fit(self, input_, epochs):
        if self.is_compiled == False:
            raise ValueError("Model is not compiled yet")
        
        if len(input_.shape) == 2:
            input_ = np.expand_dims(input_, axis=0)
        
        int_dict = {i: [] for i in range(len(self.layers))}

        
        for i in range(epochs):
            output = []  
            for j in range(len(self.layers)):
                layer = self.layers[j]
                output = layer.feedForward(input_)
                int_dict[j] = output
                input_ = output

            for j in range(len(self.layers)-1, -1, -1):
                layer = self.layers[j]
                print(layer.output)

            for j in range(len(self.layers)-1, -1, -1):
                layer = self.layers[j]
                # TODO: Menghitung turunan berdasarkan loss function yang digunakan
                if (j == len(self.layers)-1):
                    temp = layer.output
                    temp[9] = -(1-temp[9])
                    layer.dOutput = temp
                    dE_dW = multiply_arrays(layer.dOutput, layer.prev_layer.output)
                    

                    # self.dOutput = layer.output
                else:
                    dE_dW = layer.backprop()

                print("LAYER:", j)
                print(dE_dW)
                # print(dE_dW.T)

            

                

        # return output

    def load_model(self):
        ## TODO :  load model nya masukin ke layers
        return 0

    def save_model(self):
        ## TODO : save layers nya ke file
        return 0

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
        # self.filter = np.ones((self.filter_num,self.filter_size[0], self.filter_size[1]))
        # self.filter = np.random.random((self.filter_num,self.filter_size[0], self.filter_size[1]))
        self.filter = np.array([[
            [1,2,3],
            [4,7,5],
            [3,-32,25]
        ],[
            [12,18,12],
            [18,-74,45],
            [-92,45,-18]
        ]])
        print("FILTER")
        print(self.filter)
        

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
                    output[f, i, j] += np.sum(conv_region * self.filter[f])
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
    
    def backprop(self):
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
                    print("output:", input_[c,i,j])
                    output[c,i,j] = DetectorLayer.reLu(input_[c,i,j])
        self.output = output
        return output
    
    def reLu(x):
        return max(0,x)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
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
                    print("INPUT:")
                    print(input_[
                        ch,
                        i*self.stride[0]:(i+1)*self.filter_size[0],
                        j*self.stride[1]:(j+1)*self.filter_size[1]
                        ])
                    output[ch,i,j], row, col = self.pooling(input_[
                        ch,
                        i*self.stride[0]:(i+1)*self.filter_size[0],
                        j*self.stride[1]:(j+1)*self.filter_size[1]
                        ])
                    self.max_pool_index.append((row, col))
        self.output = output
        # print("Pooling output:", output[0][0])
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


    def feedForward(self, input_):
        # input_ = np.append(input_)
        self.output_no_activation = np.dot(input_, self.weights) 
        if self.activation == "relu":
            self.output = np.array(list(map(lambda x: DetectorLayer.reLu(x),np.dot(input_, self.weights))))
            return np.array(list(map(lambda x: DetectorLayer.reLu(x),np.dot(input_, self.weights))))
        elif self.activation == "sigmoid":
            self.output = np.array(list(map(lambda x: DetectorLayer.sigmoid(x),np.dot(input_, self.weights))))
            return np.array(list(map(lambda x: DetectorLayer.sigmoid(x),np.dot(input_, self.weights))))
        elif self.activation == "softmax":
            self.output = DetectorLayer.softmax(np.dot(input_, self.weights))
            return  DetectorLayer.softmax(np.dot(input_, self.weights))
            # return np.array(list(map(lambda x: DetectorLayer.softmax(x),[np.dot(input_, self.weights)]))) 

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

    def backprop(self):
        self.dWn = multiply_arrays(self.dE_dO(), self.dO_dW())
        return self.dWn

    def compile(self, prev_layer_, next_layer=None):
        if next_layer != None:
            self.next_layer = next_layer
        self.prev_layer = prev_layer_
        if len(prev_layer_.feeding_shape) != 1:
            raise ValueError("Invalid input shape for Dense Layer")
        if len(self.weights) == 0:
             self.weights = np.random.rand(prev_layer_.feeding_shape[0] + 1, self.units)
        self.feeding_shape = (self.units,)