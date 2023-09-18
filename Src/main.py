from utils import *
from CNN import *


import os




images = readDatasets()

# TODO: Change this into loadData for multiple input (for milestone 2)
for i in range(len(images)):
    print("Image:", images[i])
    preprocess_mat = createMatrix(images, i)
    model = CNN(input_shape_=preprocess_mat.shape)
    model.add(ConvolutionLayer(4, [3,3], 2, (2,2)))
    model.add(DetectorLayer())
    model.add(ConvolutionLayer(8, [5,5], 3, (2,2)))
    model.add(DetectorLayer())
    model.add(PoolingLayer((2,2), "max"))
    model.add(FlattenLayer())
    model.add(DenseLayer(8, "relu"))
    model.add(DenseLayer(1, "sigmoid"))
    model.compile()
    output = model.predict(preprocess_mat)
    print("output value", output)
    # if output > 0.5 == Bear class
    # else Panda class
    if (output[0] > 0.5):
        print("Predicted class: Bear")
    else:
        print("Predicted class: Panda")
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

