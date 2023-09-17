from Src.CNN import *
from matrix_samples import *

def test_standard3x3x1_flatten():
    model = CNN()
    model.add(FlattenLayer())
    model.compile()
    output = model.predict(standard3x3x1)
    assert (output == standard3x3x1_flatten).all()