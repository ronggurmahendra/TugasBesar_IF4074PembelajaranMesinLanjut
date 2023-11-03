import numpy as np

from lstm import LSTMLayer 

input_dummy = np.array([
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5]
])

model = LSTMLayer(10, [4,5])

model.compile(prev_layer=None, next_layer=None)

model.feedForward(input_dummy)