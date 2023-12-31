from CNN import *
from matrix_samples import *
import pytest
@pytest.mark.skip(reason="no way of currently testing this")
def test_standard3x3x1_flatten():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(FlattenLayer())
    model.compile()
    output = model.predict(standard3x3x1)
    assert (output == standard3x3x1_flatten).all()