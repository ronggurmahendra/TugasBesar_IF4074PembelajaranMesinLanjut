import numpy as np
from activation import sigmoid, tanh


class LSTMLayer():  
  n_cell = 0 
  next_layer = None
  weights = None
  biases = None
  input_shape = None # [timestamp, inputshape(1) ]

  def __init__(self, n_cell ,input_shape = None) -> None:
    if input_shape is not None:
      self.input_shape = input_shape
    self.n_cell = n_cell

  def compile(self, prev_layer, next_layer=None):
    self.prev_layer = prev_layer
    if self.prev_layer is not None:
      self.input_shape = self.prev_layer.feeding_shape
    self.weights["forget"]["U"] = np.ones(self.n_cell, self.input_shape[1])
    self.weights["forget"]["W"] = np.ones(self.n_cell, self.input_shape[1])

    self.weights["input"]["U"] = np.ones(self.n_cell, self.input_shape[1])
    self.weights["input"]["W"] = np.ones(self.n_cell, self.input_shape[1])
    
    self.weights["output"]["U"] = np.ones(self.n_cell, self.input_shape[1])
    self.weights["output"]["W"] = np.ones(self.n_cell,  self.input_shape[1])
    
    self.weights["candidate"]["U"] = np.ones(self.n_cell, self.input_shape[1])
    self.weights["candidate"]["W"] = np.ones(self.n_cell, self.input_shape[1])

    self.biases["forget"]["b"] = np.zeros(self.n_cell)
    self.biases["input"]["b"] = np.zeros(self.n_cell)
    self.biases["output"]["b"] = np.zeros(self.n_cell)
    self.biases["candidate"]["b"] = np.zeros(self.n_cell)

  def feedForward(self, x: np.ndarray) -> np.ndarray:
    # initialize the hidden state and cell state
    h = np.zeros(self.input_shape)
    c = np.zeros(self.input_shape)
    output = np.zeros(self.n_cell)
    # for each time step
    for t in range(self.n_cell):
      c = self.cell_state(x, h, c)
      h = self.hidden_state(x, self.output_gate(x, h), c)
      output[t] = h
    return output

  def candidate_gate(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    return tanh(np.dot(self.weights["candidate"]["U"], x) + np.dot(self.weights["candidate"]["W"], h_prev) + self.biases["candidate"]["b"])

  def forget_gate(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    return sigmoid(np.dot(self.weights["forget"]["U"], x) + np.dot(self.weights["forget"]["W"], h_prev) + self.biases["forget"]["b"])

  def input_gate(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    return sigmoid(np.dot(self.weights["input"]["U"], x) + np.dot(self.weights["input"]["W"], h_prev) + self.biases["input"]["b"])

  def output_gate(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    return sigmoid(np.dot(self.weights["output"]["U"], x) + np.dot(self.weights["output"]["W"], h_prev) + self.biases["output"]["b"])

  def cell_state(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> np.ndarray:
    return self.forget_gate(x, h_prev) * c_prev + self.input_gate(x, h_prev) * self.candidate_gate(x, h_prev)
  
  def hidden_state(self, x: np.ndarray, o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return o * tanh(c)