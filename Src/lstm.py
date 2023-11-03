import numpy as np
from activation import sigmoid, tanh
class LSTMLayer():  
  n_cell = 0 
  next_layer = None
  weights = None
  biases = None
  input_shape = None # [timestamp, inputshape(1) ]
  n_timestamp = 0

  def __init__(self, n_cell ,input_shape = None) -> None:
    if input_shape is not None:
      self.input_shape = input_shape
    self.n_cell = n_cell
    self.n_timestamp = input_shape[0]
    self.n_feature = input_shape[1]

  def compile(self, prev_layer=None, next_layer=None):
    self.prev_layer = prev_layer
    if self.prev_layer is not None:
      self.input_shape = self.prev_layer.feeding_shape
      self.n_timestamp = self.input_shape[0]
      self.n_feature = self.input_shape[1]

    print("(self.n_cell, self.input_shape[1])", (self.n_cell, self.input_shape[1]))
    self.weights = {
      "forget": { 
        "U": np.ones((self.n_feature, self.n_cell)),
        "W": np.ones((self.n_cell, self.n_cell)),
        "b": np.zeros(self.n_cell)
      },
      "input": {
        "U": np.ones((self.n_feature, self.n_cell)),
        "W": np.ones((self.n_cell, self.n_cell)),
        "b": np.zeros(self.n_cell)
      },
      "output": {
        "U": np.ones((self.n_feature, self.n_cell)),
        "W": np.ones((self.n_cell, self.n_cell)),
        "b": np.zeros(self.n_cell)
      },
      "candidate": {
        "U": np.ones((self.n_feature, self.n_cell)),
        "W": np.ones((self.n_cell, self.n_cell)),
        "b": np.zeros(self.n_cell)
      },
    }
    # self.weights["forget"]["U"] = np.ones((self.n_cell, self.input_shape[1]), dtype=float)
    # self.weights["forget"]["W"] = np.ones((self.n_cell, self.input_shape[1]), dtype=float)

    # self.weights["input"]["U"] = np.ones((self.n_cell, self.input_shape[1]), dtype=float)
    # self.weights["input"]["W"] = np.ones((self.n_cell, self.input_shape[1]), dtype=float)
    
    # self.weights["output"]["U"] = np.ones((self.n_cell, self.input_shape[1]), dtype=float)
    # self.weights["output"]["W"] = np.ones((self.n_cell, self.input_shape[1]), dtype=float)
    
    # self.weights["candidate"]["U"] = np.ones((self.n_cell, self.input_shape[1]), dtype=float)
    # self.weights["candidate"]["W"] = np.ones((self.n_cell, self.input_shape[1]), dtype=float)

    # self.biases["forget"]["b"] = np.zeros(self.n_cell)
    # self.biases["input"]["b"] = np.zeros(self.n_cell)
    # self.biases["output"]["b"] = np.zeros(self.n_cell)
    # self.biases["candidate"]["b"] = np.zeros(self.n_cell)

  def feedForward(self, x: np.ndarray) -> np.ndarray:
    # initialize the hidden state and cell state
    
    c = np.zeros(self.n_cell)
    h = np.zeros(self.n_cell)
    
    for t in range(self.n_timestamp):
      print("--------------", t, "--------------")
      print("x[t]: ",x[t])
      print("c : ", c)
      print("h : ", h)


      c = self.cell_state(x[t], h, c)
      # output = self.output_gate(x[t], h)
      h = self.hidden_state(x[t], self.output_gate(x[t], h), c)
    return h

    # for each time step
    # for t in range(self.n_cell):
    #   c = self.cell_state(x, h, c)
    #   h = self.hidden_state(x, self.output_gate(x, h), c)
    #   output[t] = h
    # return output

  def candidate_gate(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    return tanh(np.dot(self.weights["candidate"]["U"].T, x) + np.dot(self.weights["candidate"]["W"].T, h_prev) + self.weights["candidate"]["b"])

  def forget_gate(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    print(np.dot(self.weights["forget"]["U"].T, x))
    print('self.weights["forget"]["W"] : ', self.weights["forget"]["W"])
    print('self.weights["forget"]["U"] : ', self.weights["forget"]["U"])
    print("x", x)
    print("h_prev", h_prev)
    temp1 =  np.dot(self.weights["forget"]["U"].T, x)
    temp2 = np.dot(self.weights["forget"]["W"].T, h_prev)
    # print(temp1.shape)
    # print(temp2.shape)
    # print(self.weights["forget"]["b"].shape)
    return sigmoid(temp1 +  
                   temp2 + self.weights["forget"]["b"])

  def input_gate(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    return sigmoid(np.dot(self.weights["input"]["U"].T, x) + np.dot(self.weights["input"]["W"].T, h_prev) + self.weights["input"]["b"])

  def output_gate(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    return sigmoid(np.dot(self.weights["output"]["U"].T, x) + np.dot(self.weights["output"]["W"].T, h_prev) + self.weights["output"]["b"])

  def cell_state(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> np.ndarray:
    return self.forget_gate(x, h_prev) * c_prev + self.input_gate(x, h_prev) * self.candidate_gate(x, h_prev)
  
  def hidden_state(self, x: np.ndarray, o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return o * tanh(c)