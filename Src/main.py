from utils import *
from CNN import *
import numpy as np
from tests.matrix_samples import *

import os




# images = readDatasets()

w1 = np.array([[
            [1,2,3],
            [4,7,5],
            [3,-32,25]
        ],[
            [12,18,12],
            [18,-74,45],
            [-92,45,-18]
        ]])

w2 = np.array([
            [1,2],
            [3,-4],
        ])

w3 = np.array([
            [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.04, 0.05, 0.01],
            [0.02, 0.03, 0.03, 0.02, 0.01, 0.02, 0.07, 0.08, 0.05, 0.01],
        ])
matrix = np.array([
    [4,1,3,5,3],
    [2,1,1,2,2],
    [5,5,1,2,3],
    [2,2,4,3,2],
    [5,1,3,4,5]
]) 

print(matrix.shape)
# TODO: Change this into loadData for multiple input (for milestone 2)
for i in range(1):
    print(matrix)
    # print("Image:", images[i])
    # preprocess_mat = createMatrix(images, i)
    model = CNN(input_shape_=matrix.shape)
    model.add(ConvolutionLayer(2, [3,3], 0, (1,1)))
    model.add(DetectorLayer())
    # # model.add(ConvolutionLayer(8, [5,5], 3, (2,2)))
    # model.add(DetectorLayer())
    model.add(PoolingLayer((3,3), "max"))
    model.add(FlattenLayer())
    model.add(DenseLayer(2, "relu", w2))
    model.add(DenseLayer(10, "softmax", w3))
    # model.add(DenseLayer(1, "sigmoid"))
    model.compile()
    model.fit(matrix, 1)
    
    # output = model.predict(matrix)
    # print("output value", output)
    # # if output > 0.5 == Bear class
    # # else Panda class
    # if (output[0] > 0.5):
    #     print("Predicted class: Bear")
    # else:
    #     print("Predicted class: Panda")
    # print(output.shape)






# print()

# model = CNN(input_shape_=preprocess_mat.shape)
# model.add(ConvolutionLayer(4, [3,3], 2, (2,2)))
# model.add(DetectorLayer())
# model.add(ConvolutionLayer(8, [5,5], 3, (2,2)))
# model.add(DetectorLayer())
# model.add(PoolingLayer((2,2), "max"))
# model.add(FlattenLayer())
# model.add(DenseLayer(8, "relu"))
# model.add(DenseLayer(2, "sigmoid"))
# model.compile()
# output = model.predict(preprocess_mat)
# print(output)
# print(output.shape)

