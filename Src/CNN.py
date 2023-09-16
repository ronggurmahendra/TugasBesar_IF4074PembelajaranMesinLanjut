import numpy as np

class CNN: 
    layers = []

    def __init__(self):
        ## TODO : kek nya ada parameter input dimentionya Cmiiw
        self.layers = []
    
    def add(self, layer_):
        self.layers.append(layer_)

    def predict(self, input_):
        # add a dimension to a single channel input
        if len(input_.shape) == 2:
            input_ = np.expand_dims(input_, axis=0)

        output = []  
        for layer in self.layers:
            output = layer.feedForward(input_)
            input_ = output
        return output
    
    def fit(self):
        ## TODO : ini tahap back prop nya ngga di dalem milestone 1 CMIIW
        return 0

    def load_model(self):
        ## TODO :  load model nya masukin ke layers
        return 0

    def save_model(self):
        ## TODO : save layers nya ke file
        return 0


class ConvolutionLayer:
    filters = -1
    kernel_size = [-1,-1]
    padding = -1
    stride = -1

    def __init__(self, filters_, kernel_size_, padding_, stride_):
        ## TODO : cek validitas input >1/ dimensi terhadap input matrix(perlu di infer dari layer sebelumnya)
        self.filters = filters_
        self.kernel_size = kernel_size_
        self.padding = padding_
        self.stride = stride_
    

    def feedForward(self, input_):
        ## TODO :  masi syntax error, baru implement secara algoritmik aja
        height, width, channel = input_.shape

        conv_height = height - self.filter_size[0] + 1
        conv_width = width - self.filter_size[1] + 1

        output = np.zeros((self.filters, conv_height, conv_width)) # output feature map
        
        for f in range (self.filters):
            for i in range(conv_height):
                for j in range(conv_width):
                    ## TODO : hmmmm, perlu merenung dulu untuk disini tapi basically perhitungan convolusinya pake np.conv 
                    # output[f,i,j] += np.
                    output[f,i,j] += 1

        return 0
    
class DetectorLayer:
    def feedForward(self, input_):
        channel, height, width = input_.shape
        output = np.zeros((channel, height, width)) # output feature map
        for c in range (channel):
            for i in range(height):
                for j in range(width):
                    output[c,i,j] = self.reLu(output[c,i,j])
        return output
    
    def reLu(x):
        return max(0,x)
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
    
    def pooling(self, input_):
        if self.mode == "max":
            return np.max(input_)
        elif self.mode == "average":
            return np.average(input_)
    

class FlattenLayer:
    def feedForward(self, input_):
        return np.reshape(input_, (-1))


class DenseLayer:
    def __init__(self, units_, input_size_):
        self.units = units_
        self.weights = np.random.rand(input_size_ + 1, units_)

    def feedForward(self, input_):
        input_ = np.append(input_, 1)
        return np.dot(input_, self.weights)
