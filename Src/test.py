# for testing mainly

from CNN import * 

standard6x6x1 = np.array([[14,0,12,23,41,65],[12,23,14,0,12,23],[41,65,12,23,14,0],[12,23,14,0,12,23],[23,41,65,12,23,14],[14,0,12,23,41,65]])

model = CNN()
model.add(ConvolutionLayer(32, [3,3], 1, 1, standard6x6x1.shape))
output = model.predict(standard6x6x1)
assert output.shape == (32, 4, 4)

print("Executed...")