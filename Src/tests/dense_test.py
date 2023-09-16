from Src.CNN import *
from matrix_samples import *

def test_standard3x3x1():
    model = CNN()
    model.add(FlattenLayer())
    model.add(DenseLayer(4, 9))
    output = model.predict(standard3x3x1)
    assert output.shape == (4,)
