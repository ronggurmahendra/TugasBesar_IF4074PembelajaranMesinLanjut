import numpy as np
import json

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
    
    def fit(self, X, Y, epoch=1, batch_size=1, learning_rate=0.01, verbose=False):
        if self.is_compiled == False:
            raise ValueError("Model is not compiled yet")
        batches = np.array_split(X, len(X) / batch_size)
        output_batches = np.array_split(Y, len(Y) / batch_size)

        for i in range(epoch):
            for batch, output_batch in zip(batches, output_batches):
                output = self.predict(batch)
                accuracy = np.sum(np.argmax(output, axis=1) == np.argmax(output_batch, axis=1)) / batch_size
                error = np.sum(np.square(output_batch - output)) / batch_size
                if verbose:
                    print("Epoch: ", i, " Accuracy: ", accuracy, " Error: ", error)
                dE_dO = (output_batch - output) / batch_size
                for layer in reversed(self.layers):
                    dE_dO = layer.backprop(dE_dO, learning_rate)
            output = self.predict(batch)
            accuracy = np.sum(np.argmax(output, axis=1) == np.argmax(output_batch, axis=1)) / batch_size
            print("Epoch: ", i, " Accuracy: ", accuracy)
        print("Training finished")
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
                layer.compile(self)
            else:
                layer.compile(self.layers[i-1])
        self.is_compiled = True


class ConvolutionLayer:
    filter_num = -1
    filter_size = [-1,-1]
    filter = [[]]
    bias = 1
    padding = -1
    stride = -1
    input_shape = [-1,-1,-1]

    def __init__(self, filter_num_, filter_size_, padding_, stride_, input_shape_ = [-1,-1]):
        self.filter_num = filter_num_
        self.filter_size = filter_size_
        self.padding = padding_
        self.stride = stride_
        self.input_shape = input_shape_
        # self.filter = np.ones((self.filter_num,self.filter_size[0], self.filter_size[1]))
        self.filter = np.random.random((self.filter_num,self.filter_size[0], self.filter_size[1]))
        

    def feedForward(self, input_):
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
        return output
    def compile(self, prev_layer_):
        self.input_shape = prev_layer_.feeding_shape
        input_height_padded = prev_layer_.feeding_shape[1] + 2 * self.padding
        input_width_padded = prev_layer_.feeding_shape[2] + 2 * self.padding
        if len(prev_layer_.feeding_shape) != 3:
            raise ValueError("Invalid input shape for Convolution Layer")
        self.feeding_shape = (self.filter_num, (input_height_padded - self.filter_size[0] + 1) // self.stride[0], (input_width_padded - self.filter_size[1] + 1) // self.stride[1])

class DetectorLayer:
    def feedForward(self, input_):
        channel, height, width = input_.shape
        output = np.zeros((channel, height, width)) # output feature map
        for c in range (channel):
            for i in range(height):
                for j in range(width):
                    output[c,i,j] = DetectorLayer.reLu(output[c,i,j])
        return output
    
    def reLu(x):
        return np.maximum(0,x)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def reLu_derivative(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def sigmoid_derivative(x):
        return DetectorLayer.sigmoid(x) * (1 - DetectorLayer.sigmoid(x))
    def compile(self, prev_layer_):
        self.feeding_shape = prev_layer_.feeding_shape

class PoolingLayer:
    kernel_size = [-1,-1]
    stride = 0 
    mode = "" # either max / average

    def __init__(self, filter_size_, mode_, stride_=None):
        self.filter_size = filter_size_
        # definition of stride is define as
        # https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
        if stride_ is None:
            self.stride = filter_size_
        self.mode = mode_

    def feedForward(self, input_):
        channel, input_height, input_width = input_.shape
        output_height = input_height // self.stride[0]
        output_width = input_width // self.stride[1]
        
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Invalid input or filter size")

        output = np.zeros((channel, output_height, output_width))
        for ch in range(channel):
            for i in range(output_height):
                for j in range(output_width):
                    output[ch,i,j] = self.pooling(input_[
                        ch,
                        i*self.stride[0]:(i+1)*self.filter_size[0],
                        j*self.stride[1]:(j+1)*self.filter_size[1]
                        ])
        return output
    
    def compile(self, prev_layer_):
        channel, input_height, input_width = prev_layer_.feeding_shape
        self.feeding_shape = (channel, input_height // self.stride[0], input_width // self.stride[1])
    
    def pooling(self, input_):
        if self.mode == "max":
            return np.max(input_)
        elif self.mode == "average":
            return np.average(input_)

class FlattenLayer:
    def feedForward(self, input_):
        return np.array([np.reshape(d, (-1)) for d in input_])
    
    def backprop(self, dE_dO, learning_rate):
        return dE_dO

    def compile(self, prev_layer_):
        total = 1
        for x in prev_layer_.feeding_shape:
            total *= x
        self.feeding_shape = (total,)

class DenseLayer:
    def __init__(self, units_, activation_):
        self.units = units_
        self.weights = []
        self.bias = []
        self.activation = activation_
        self.output = []
        self.input = []
    def feedForward(self, batch_input):
        # add a dimension to input_ because it is seen as a batch
        if len(batch_input.shape) == 1:
            batch_input = np.expand_dims(batch_input, axis=0)
        self.input = batch_input
        if self.activation == "relu":
            self.output = np.asarray([DetectorLayer.reLu(np.dot(d, self.weights) + self.bias) for d in batch_input])
        elif self.activation == "sigmoid":
            self.output = np.asarray([np.array(list(map(lambda x: DetectorLayer.sigmoid(x),np.dot(d, self.weights) + self.bias))) for d in batch_input])

        return self.output
    # backprop is performed in each batch
    def backprop(self, dE_dO, learning_rate):
        dE_dO = np.array(dE_dO)
        dO_dN = DetectorLayer.reLu_derivative(self.output) if self.activation == "relu" else DetectorLayer.sigmoid_derivative(self.output)
        dN_dW = self.input
        dE_dN = (dE_dO * dO_dN)
        dE_dW = np.zeros(self.weights.shape)
        for i in range(len(dN_dW)):
            dE_dW += (np.outer(dN_dW[i], dE_dN[i]) / len(dE_dN))
        self.weights -= learning_rate * dE_dW
        self.bias -= learning_rate * np.sum(dE_dN) / len(dE_dN)

        dN_dI = self.weights
        dE_dI = np.array(list(map(lambda x: x * dN_dI, dE_dN)))
        return np.dot(dE_dI, self.weights.T)
    def compile(self, prev_layer_):
        if len(prev_layer_.feeding_shape) > 2:
            raise ValueError("Invalid input shape for Dense Layer")
        # self.weights = np.random.rand(prev_layer_.feeding_shape[0] + 1, self.units)
        self.weights = np.ones((prev_layer_.feeding_shape[0], self.units))
        self.bias = np.ones((self.units))
        self.feeding_shape = (self.units,)