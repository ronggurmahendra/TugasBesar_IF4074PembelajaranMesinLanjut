import numpy as np
verbose = True
class CNN: 

    def __init__(self, input_size_ = None):
        ## TODO : kek nya ada parameter input dimentionya Cmiiw
        self.layers = []
        self.is_compiled = False
        self.input_size = input_size_
        self.feeding_shape = input_size_
    
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
    
    def fit(self):
        ## TODO : ini tahap back prop nya ngga di dalem milestone 1 CMIIW
        return 0

    def load_model(self):
        ## TODO :  load model nya masukin ke layers
        return 0

    def save_model(self):
        ## TODO : save layers nya ke file
        return 0

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
    padding = -1
    stride = -1
    input_shape = [-1,-1,-1]

    def __init__(self, filter_num_, filter_size_, padding_, stride_, input_shape_ = [-1,-1]):
        ## TODO : cek validitas input >1/ dimensi terhadap input matrix(perlu di infer dari layer sebelumnya)
        self.filter_num = filter_num_
        self.filter_size = filter_size_
        self.padding = padding_
        self.stride = stride_
        self.input_shape = input_shape_
        # self.filter = np.ones((self.filter_num,self.filter_size[0], self.filter_size[1]))
        # self.filter = np.random.randint(1,10,size = (self.filter_num,self.filter_size[0], self.filter_size[1]))
        self.filter = np.random.random((self.filter_num,self.filter_size[0], self.filter_size[1]))
        

    def feedForward(self, input_):
        ## TODO :  masi syntax error, baru implement secara algoritmik aja
        # padding 
        # input_padded = np.pad(input_,self.padding, 'constant', constant_values=0)
        # pake np.pad nge pad channelnya juga
        input_height_padded = input_.shape[1] + 2 * self.padding
        input_width_padded = input_.shape[2] + 2 * self.padding
        
        input_padded = np.zeros((input_.shape[0], input_height_padded, input_width_padded))
        input_padded[ : , 
               self.padding: self.padding + input_.shape[1],
               self.padding: self.padding + input_.shape[2]] = input_

        
        print("input_padded : ", input_padded)
        channel, height, width = input_padded.shape

        print("filter : ", self.filter)
        print("filter_size : ", self.filter_size)
        print("input_padded.shape", input_padded.shape)
        conv_height = (height - self.filter_size[0] + 1) // self.stride[0]
        conv_width = (width - self.filter_size[1] + 1) // self.stride[1]
    
        print("self.filter_num, conv_height, conv_width = ", self.filter_num, conv_height, conv_width)
        output = np.zeros((self.filter_num, conv_height, conv_width)) # output feature map
        
        print("output.shape", output.shape)
        for f in range (self.filter_num):
            for i in range(conv_height):
                for j in range(conv_width):
                    ## TODO : ini sepemahaman aku mungkin salah, perlu di test 
                    conv_region = input_padded[:,i*self.stride[0]:i*self.stride[0] + self.filter_size[0], j*self.stride[1]:j*self.stride[1] + self.filter_size[1]]
                    output[f, i, j] += np.sum(conv_region * self.filter[f])
        print("output : ", output)
        print("output.shape : ", output.shape)
        return output
    def compile(self, prev_layer_):
        pass
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
        return max(0,x)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def compile(self, prev_layer_):
        pass
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
        pass
    
    def pooling(self, input_):
        if self.mode == "max":
            return np.max(input_)
        elif self.mode == "average":
            return np.average(input_)

class FlattenLayer:
    def feedForward(self, input_):
        return np.reshape(input_, (-1))
    
    def compile(self, prev_layer_):
        total = 1
        for x in prev_layer_.feeding_shape:
            total *= x
        self.feeding_shape = (total,)


class DenseLayer:
    def __init__(self, units_, activation_):
        self.units = units_
        self.weights = []
        self.activation = activation_

    def feedForward(self, input_):
        input_ = np.append(input_, 1)
        if self.activation == "relu":
            return np.array(list(map(lambda x: DetectorLayer.reLu(x),np.dot(input_, self.weights))))
        elif self.activation == "sigmoid":
            return np.array(list(map(lambda x: DetectorLayer.sigmoid(x),np.dot(input_, self.weights))))

    def compile(self, prev_layer_):
        if len(prev_layer_.feeding_shape) != 1:
            raise ValueError("Invalid input shape for Dense Layer")
        self.weights = np.random.rand(prev_layer_.feeding_shape[0] + 1, self.units)