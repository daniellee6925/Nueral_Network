x = [1.0, -2.0, 3.0]  # inputs
w = [-3.0, -1.0, 2.0]  # weights
b = 1.0  # bias

# multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# add all weights plus bias
z = xw0 + xw1 + xw2 + b


# RELU activation function
y = max(z, 0)

# backward pass
# derivative value from next layer
dvalue = 1

# derivative of the Relu Activation Function and Chain Rule
drelu_dz = dvalue * (1.0 if z > 0 else 0.0)

# value of the partial derivative of the sum operation
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1

# partial derivatove of ReLU w.r.t. first weight input wx0
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db


# function is X * weight.
# partial derivative of x * weight w.r.t to x is weight: w[0]
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
# partial derivative of x * weight w.r.t to weight is x: x[0]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

# derivative for ReLU(x * weight) = dRelu x partial derivative of (x * weight) - Chain Rule
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw0 * dmul_dw1
drelu_dw2 = drelu_dxw0 * dmul_dw2

# combine together
drelu_dz = dvalue * (1.0 if z > 0 else 0.0) * w[0]

# gradients
dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db


import numpy as np

# an array of  incremental gradient values
dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

# input values
inputs = np.array([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]])

# 3 sets of weights for each neuron. 4 inputs -> 4 weights
# keep weights transposed for calculation with input
weights = np.array(
    [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
).T

# bias for each nueron
biases = np.array([[2, 3, 0.5]])

# Forward Pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

# Backpropagation
# derivative of the relu function
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# chain rule - partial derivatives of inputs, weights, biases
dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

# update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases
