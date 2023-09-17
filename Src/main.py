from utils import *
from CNN import *


images = readDatasets()

# TODO: Change this into loadData for multiple input (for milestone 2)
preprocess_mat = createMatrix(images, 0)




print(preprocess_mat)

model = CNN()
model.add(ConvolutionLayer(32, [3,3], 0, (0,1)))
model.add(DetectorLayer())
model.add(PoolingLayer((2,2), "max"))

output = model.predict(preprocess_mat)

