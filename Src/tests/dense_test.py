from CNN import *
from matrix_samples import *
import pytest

@pytest.mark.skip(reason="no way of currently testing this")
def test_standard3x3x1():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(FlattenLayer())
    model.add(DenseLayer(4, "relu"))
    model.compile()
    output = model.predict(standard3x3x1)
    assert output.shape == (1,4)

@pytest.mark.skip(reason="no way of currently testing this")
def test_standard3x3x1x2():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(FlattenLayer())
    model.add(DenseLayer(4, "relu"))
    model.compile()
    output = model.predict(standard3x3x1x2)
    assert output.shape == (2,4)

@pytest.mark.skip(reason="no way of currently testing this")
def test_backprop_standard3x3x1x2():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(FlattenLayer())
    model.add(DenseLayer(4, "relu"))
    model.compile()
    output = model.predict(standard3x3x1x2)
    model.fit(X=standard3x3x1x2, Y=output_standard3x3x1x2,epoch=1,batch_size=1,learning_rate=0.01)
    output = model.predict(standard3x3x1x2)
    assert output.shape == output_standard3x3x1x2.shape

matrix = np.array([
    [4,1,3,5,3],
    [2,1,1,2,2],
    [5,5,1,2,3],
    [2,2,4,3,2],
    [5,1,3,4,5]
]) 

@pytest.mark.skip(reason="no way of currently testing this")
def test_standard3x3x2():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(FlattenLayer())
    model.add(DenseLayer(4, "relu"))
    model.compile()
    output = model.predict(standard3x3x2)
    assert output.shape == (4,2)

def test_standard3x3x2_with_pooling():
    model = CNN(input_shape_=matrix.shape)
    model.add(ConvolutionLayer(2, [3,3], 0, (1,1)))
    model.add(PoolingLayer((2,2), "max"))
    model.add(FlattenLayer())
    model.add(DenseLayer(4, "relu"))
    model.compile()
    output = model.predict(matrix)
    model.fit(matrix, Y=[1], epochs=2)
    assert output.shape == (1,4)

def test_standard3x3x2_with_pooling_mse():
    model = CNN(input_shape_=matrix.shape)
    model.add(ConvolutionLayer(2, [3,3], 0, (1,1)))
    model.add(PoolingLayer((2,2), "max"))
    model.add(FlattenLayer())
    model.add(DenseLayer(1, "sigmoid"))
    model.compile()
    output = model.predict(matrix)
    model.fit(matrix, Y=[1], epochs=2, loss="mse")
    assert output.shape == (1,)