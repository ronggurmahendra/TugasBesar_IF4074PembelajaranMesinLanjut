import numpy as np

from utils import multiply_and_sum
# # Initialize an empty dictionary
# int_dict = {}

# # List of keys (integers)
# keys = [1, 2, 3]

# # Create the dictionary with empty arrays as values for each key
# int_dict = {key: [] for key in keys}

# # Print the initial dictionary
# print(int_dict)

# # Update values for a specific key (e.g., key=1)
# # int_dict[1].append([100, 50])
# int_dict[1] = [100, 50]
# # int_dict[1].append(20)
# # int_dict[1].append(30)

# # Print the updated dictionary
# print(int_dict)
def softmax(x):
        print(x)
        exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
        return exp_x / exp_x.sum(axis=0, keepdims=True)

x = np.array([ 9.85592366e-01,  1.42000125e-02,  2.04587985e-04,  2.94762018e-06,
  4.24681085e-08,  6.11863173e-10,  8.81547486e-12,  6.11863173e-10,
  4.24681085e-08, -1.00000000e+00])
w3 = np.array([
            [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.04, 0.05, 0.01],
            [0.02, 0.03, 0.03, 0.02, 0.01, 0.02, 0.07, 0.08, 0.05, 0.01],
        ])
print(multiply_and_sum(x, w3))
# # print(x)
# output = softmax(x)
# print("Softmax output:", output)