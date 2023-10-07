from CNN import *
from matrix_samples import *
import pytest

@pytest.mark.skip(reason="no way of currently testing this")
def test_standard6x6x1():
    model = CNN(input_shape_=standard6x6x1.shape)
    model.add(ConvolutionLayer(32, [3,3], 0, (1,1)))
    model.compile()
    output = model.predict(standard6x6x1)
    assert output.shape == (32, 4, 4)
@pytest.mark.skip(reason="no way of currently testing this")
def test_standard6x6x1_with_padding():
    model = CNN(input_shape_=standard6x6x1.shape)
    model.add(ConvolutionLayer(32, [3,3], 1, (1,1)))
    model.compile()
    output = model.predict(standard6x6x1)
    assert output.shape == (32, 6, 6)
@pytest.mark.skip(reason="no way of currently testing this")
def test_standard6x6x1_with_stride():
    model = CNN(input_shape_=standard6x6x1.shape)
    model.add(ConvolutionLayer(32, [3,3], 0, (2,2)))
    model.compile()
    output = model.predict(standard6x6x1)
    assert output.shape == (32, 2, 2)

