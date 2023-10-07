# for testing mainly

from CNN import * 

# standard6x6x1 = np.array([[14,0,12,23,41,65],[12,23,14,0,12,23],[41,65,12,23,14,0],[12,23,14,0,12,23],[23,41,65,12,23,14],[14,0,12,23,41,65]])
standard6x6x1 = np.array([[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2]])
standard6x6x2 = np.array([[[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2]],[[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2],[2,2,2,2,2,2]]])

model = CNN(input_shape_=standard6x6x1.shape)
model.add(ConvolutionLayer(32, [3,3], 0, (1,1)))
model.add(DetectorLayer())
model.add(PoolingLayer((2,2), "max"))
model.compile()

model.save_model("model.json")
output = model.predict(standard6x6x2)

loaded_model = CNN(input_shape_=standard6x6x1.shape)
loaded_model.load_model("model.json")
loaded_model.compile()

output = loaded_model.predict(standard6x6x1)
print(output)
print("Executed...")