from CNN import *
from matrix_samples import *
import pytest
@pytest.mark.skip(reason="no way of currently testing this")
def test_standard3x3x1_no_dense():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(ConvolutionLayer(32, [3,3], 0, (1,1)))
    model.add(DetectorLayer())
    model.add(PoolingLayer((2,2), "max"))
    model.compile()
    output = model.predict(standard6x6x1)
    assert output.shape == (32, 2, 2)
