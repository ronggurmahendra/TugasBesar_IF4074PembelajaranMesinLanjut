# for testing mainly

from CNN import * 

# standard6x6x1 = np.array([[14,0,12,23,41,65],[12,23,14,0,12,23],[41,65,12,23,14,0],[12,23,14,0,12,23],[23,41,65,12,23,14],[14,0,12,23,41,65]])
standard6x6x1 = np.array([[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2]])

model = CNN()
model.add(ConvolutionLayer(32, [3,3], 2, (2,2), standard6x6x1.shape))
model.compile()
output = model.predict(standard6x6x1)

print("Executed...")