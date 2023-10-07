from Src.CNN import *
from matrix_samples import *

def test_standard3x3x1():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(FlattenLayer())
    model.add(DenseLayer(4, "relu"))
    model.compile()
    output = model.predict(standard3x3x1)
    assert output.shape == (1,4)

def test_standard3x3x1x2():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(FlattenLayer())
    model.add(DenseLayer(4, "relu"))
    model.compile()
    output = model.predict(standard3x3x1x2)
    print(output)
    assert output.shape == (2,4)

def test_backprop_standard3x3x1x2():
    model = CNN(input_shape_=standard3x3x1.shape)
    model.add(FlattenLayer())
    model.add(DenseLayer(4, "relu"))
    model.compile()
    output = model.predict(standard3x3x1x2)
    model.fit(X=standard3x3x1x2, Y=output_standard3x3x1x2,epoch=1,batch_size=1,learning_rate=0.01)
    output = model.predict(standard3x3x1x2)
    assert output.shape == output_standard3x3x1x2.shape
