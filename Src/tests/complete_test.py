# from Src.CNN import *
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(project_root)
sys.path.append(project_root)


from Src.CNN import *
from matrix_samples import *

def test_standard3x3x1_no_dense():
    model = CNN()
    model.add(ConvolutionLayer(32, [3,3], 0, (0,1)))
    model.add(DetectorLayer())
    model.add(PoolingLayer((2,2), "max"))
    output = model.predict(standard6x6x1)
    print(output)
    # assert output.shape == (32, 2, 2)

test_standard3x3x1_no_dense()