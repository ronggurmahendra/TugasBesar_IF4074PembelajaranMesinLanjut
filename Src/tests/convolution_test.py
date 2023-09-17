from Src.CNN import *
from matrix_samples import *

def test_standard6x6x1():
    model = CNN()
    model.add(ConvolutionLayer(32, [3,3], 0, (0,1)))
    model.compile()
    output = model.predict(standard6x6x1)
    assert output.shape == (32, 4, 4)

def test_standard6x6x1_with_padding():
    model = CNN()
    model.add(ConvolutionLayer(32, [3,3], 1, (0,1)))
    model.compile()
    output = model.predict(standard6x6x1)
    assert output.shape == (32, 2, 2)

def test_standard6x6x1_with_stride():
    model = CNN()
    model.add(ConvolutionLayer(32, [3,3], 0, (0,2)))
    model.compile()
    output = model.predict(standard6x6x1)
    assert output.shape == (32, 2, 2)

