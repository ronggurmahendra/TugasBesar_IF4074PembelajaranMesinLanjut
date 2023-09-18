from Src.CNN import *
from matrix_samples import *

def test_standard3x3x1():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(FlattenLayer())
    model.add(DenseLayer(4, "relu"))
    model.compile()
    output = model.predict(standard3x3x1)
    assert output.shape == (4,)
