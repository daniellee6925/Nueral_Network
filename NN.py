import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

np.random.seed(0)

"""
def create_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, points)
            + np.random.randn(points) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y
    


import matplotlib.pyplot as plt

print("here")

X, y = create_data(100, 3)

plt.scatter(X[:, 0], X[:, 1])
plt.show()


plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()
"""

"""
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(outputs)
"""


class Layer_Dense:
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass
    def backward(self, dvalues):
        # gradients on paremeters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    # forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        # zero gradient where inputs are negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )  # prevent overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    # calculates the data losses
    def calculate(self, output, y):
        # calculate sampmle losses
        sample_losses = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_losses)
        # return loss
        return data_loss


# corss-entropy loss - child of parent class loss
class Loss_CategoricalCrossEntropy(Loss):
    # forward pass
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clip data to prevent divison by 0
        # clip both sides to not drag mean
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # prevent neg INF

        # for categorical values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # for one-hot encoded values
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # one hot encode labels
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # derivative of cross entropy
        self.dinputs = -y_true / dvalues

        # normalize gradient
        self.dinputs = self.dinputs / samples


dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

# calculate values along first axis
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print("acc: ", accuracy)

""" 
layer_outputs = []  # output of current layer
for neuron_weights, nueron_bias in zip(weights, biases):
    nueron_output = 0  # output of given neuron
    for n_input, weight in zip(inputs, neuron_weights): #loop through elements within each array
        nueron_output += n_input * weight  # input x weight
    nueron_output += nueron_bias  # add bias
    layer_outputs.append(nueron_output)

print(layer_outputs)
"""
