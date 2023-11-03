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


import LSTMLayer from lstm

model = LSTMLayer(10, [4,5])

model.compile()