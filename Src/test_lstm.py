import numpy as np

from lstm import LSTMLayer 
import numpy as np
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    data_len = len(data)
    for i in range(data_len - seq_length):
        seq_end = i + seq_length
        seq_x = data[i:seq_end]
        seq_y = data[seq_end]
        sequences.append(seq_x)
        targets.append(seq_y)
    return np.array(sequences), np.array(targets)
# 12345678910
# 12345 6
# 23456 7
seq_length = 5
seq_dummy = [i for i in range(1, 100 + 1)]
x, y = create_sequences(seq_dummy,seq_length)
print(x.shape)
print(y.shape)
input_dummy = np.array([
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5]
])

model = LSTMLayer(10, [seq_length,1])

model.compile(prev_layer=None, next_layer=None)
# model.feedForward(x)

for i in x:
    model.feedForward(i)