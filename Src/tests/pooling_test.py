from Src.CNN import *
from matrix_samples import *

def test_standard3x3x1_with_2x2_maxpooling():
    model = CNN()
    model.add(PoolingLayer((2,2), 0, "max"))
    output = model.predict(standard3x3x1)
    assert (output == standard3x3x1_maxpooled2x2).all()
def test_standard3x3x1_with_2x2_avgpooling():
    model = CNN()
    model.add(PoolingLayer((2,2), 0, "average"))
    output = model.predict(standard3x3x1)
    assert (output == standard3x3x1_avgpooled2x2).all()
