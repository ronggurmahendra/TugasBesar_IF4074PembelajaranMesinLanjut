# for testing mainly

from CNN import * 

model = CNN()
model.add(ConvolutionLayer(32, [3,3], 1, 1))

print("Done")