import numpy as np

class CNN: 
    layers = []

    def __init__(self):
        ## TODO : kek nya ada parameter input dimentionya Cmiiw
        self.layers = []
    
    def add(self, layer_):
        self.layers.append(layer_)

    def predict(self, input_):
        ## TODO : di test
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
    

class PoolingLayer:
    kernel_size = [-1,-1]
    stride = 0 
    mode = "" # either max / average

    def __init__(self, kernel_size_, padding_, mode_):
        ## TODO : cek validitas input >1/ dimensi terhadap input matrix(perlu di infer dari layer sebelumnya)
        self.kernel_size = kernel_size_
        self.stride = stride_
        self.mode = mode_
    

class FlattenLayer:
    def __init__(self):
        # TODO : ini perlu ngga ya?
        print("initializing flatten layer")

    def feedForward(self, input_):
        # TODO : implement flatten
        return 0
