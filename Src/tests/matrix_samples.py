import numpy as np

standard3x3x1 = np.array([[14,0,12],[23,41,65],[12,23,14]])
standard3x3x1_flatten = np.array([14,0,12,23,41,65,12,23,14])
standard3x3x1_maxpooled2x2 = [[41]]
standard3x3x1_avgpooled2x2 = [[19.5]]

standard3x3x1x2 = np.array([[[14,0,12],[23,41,65],[12,23,14]],[[14,0,12],[23,41,65],[12,23,14]]])
output_standard3x3x1x2 = np.array([[1,1,1,1],[1,1,1,1]])

standard6x6x1 = np.array([[14,0,12,23,41,65],[12,23,14,0,12,23],[41,65,12,23,14,0],[12,23,14,0,12,23],[23,41,65,12,23,14],[14,0,12,23,41,65]])
