import numpy as np

class CNN: 
    layers = []

    def __init__(self):
        ## TODO :  
        self.layers = []
    
    def add(self, layer_):
        self.layers.append(layer_)

    def predict(self):
        ## TODO :  
        return 0
    
    def fit(self):
        ## TODO :  
        return 0

    def load_model(self):
        ## TODO :  
        return 0

    def save_model(self):
        ## TODO :  
        return 0


class ConvolutionLayer:
    filters = -1
    kernel_size = [-1,-1]
    padding = -1
    stride = -1

    def __init__(self, filters_, kernel_size_, padding_, stride_):
        ## TODO : cek validitas input >1/ dimensi terhadap input matrix(perlu di infer dari layer sebelumnya)
        filters = filters_
        kernel_size = kernel_size_
        padding = padding_
        stride = stride_
    

    def FeedForward(self, input_):
        ## TODO :  masi syntax error, baru implement secara algoritmik aja
        height, width, channel = input_.shape

        conv_height = height - self.filter_size[0] + 1
        conv_width = width - self.filter_size[1] + 1

        output = np.zeros((self.filters, conv_height, conv_width)) # output feature map
        
        for f in range (self.filters):
            for i in range(conv_height):
                for j in range(conv_width):
                    ## TODO : hmmmm, perlu merenung dulu untuk disini tapi basically pake np.conv 
                    output[f,i,j] += np.

        return 0
    

class PoolingLayer:
    kernel_size = []
    stride = 0 
    mode = "" # either max / average