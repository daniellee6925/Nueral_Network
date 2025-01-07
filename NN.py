import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


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


class Layer_Dense:
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs
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
        self.inputs = inputs
        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )  # prevent overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # create an empty array (which will become the resulting gradient array)
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            # flatten output array
            single_output = single_output.reshape(-1, 1)
            # calculate Jacobian Matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            # calculate sample wise graident
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    # calculates the data losses
    def calculate(self, output, y):
        # calculate sampmle losses usig categorical cross entropy
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
    def backward(self, dvalues, y_true):  # dvalues = y_pred
        # number of samples
        samples = len(dvalues)
        # number of labls in each sample
        labels = len(dvalues[0])

        # one hot encode labels if labels are sparse
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # derivative of cross entropy
        self.dinputs = -y_true / dvalues

        # normalize gradient
        self.dinputs = self.dinputs / samples


# combined softmax + cross-entropy loss for faster backward step
class Activation_Softmax_loss_CategoricalCrossentropy:
    def __init__(self):
        # set activation to Softmax
        self.activation = Activation_Softmax()

        # set loss to categorical cross entropy
        self.loss = Loss_CategoricalCrossEntropy()

    # forward pass
    def forward(self, inputs, y_true):
        # inputs foward using the activation function
        self.activation.forward(inputs)
        # set the output
        self.output = self.activation.output
        # calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        # if labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy dvalues
        self.dinputs = dvalues.copy()

        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        # normalize gradient
        self.dinputs = self.dinputs / samples


# stochastic gradient descent
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:  # decay rate not zero
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    # update parameters
    def update_params(self, layer):
        if self.momentum:
            # if layer doesn't contain momentum arrays, create them with zeros
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # build weight updates with momentum
            # take previous updates multiplied by retain factor and udpate with current gradient
            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * layer.dweights
            )
            # set current weight updates to momentums for next update
            layer.weight_momentums = weight_updates

            # do the same for biases
            biase_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )
            # set current weight updates to momentums for next update
            layer.bias_momentums = biase_updates

        # Vanilla SGD
        else:
            weight_updates = -self.learning_rate * layer.dweights
            biase_updates = -self.learning_rate * layer.dbiases

        # update weights and biases
        layer.weights += weight_updates
        layer.biases += biase_updates

    # call once before any parameter updates
    def post_update_params(self):
        # update iterations
        self.iterations += 1


# Adagrad Optimizer
class Optimizer_Adagrad:
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:  # decay rate not zero
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    # update parameters
    def update_params(self, layer):
        # if layer doesn't contain cache arrays, fill them with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * layer.dbiases
            / (np.sqrt(layer.bais_cache) + self.epsilon)
        )

    # call once before any parameter updates
    def post_update_params(self):
        # update iterations
        self.iterations += 1


if __name__ == "__main__":
    # create data set
    X, y = spiral_data(100, 3)
    np.random.seed(0)

    # create first dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 64)

    # create RELU activation
    relu_activation = Activation_ReLU()

    # create 2nd dense layer with 3 input features (same as output num. from first layer) and 3 output values
    dense2 = Layer_Dense(64, 3)

    # create softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_loss_CategoricalCrossentropy()

    # create optimizer object
    # optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)
    optimizer = Optimizer_Adagrad(decay=1e-4)

    for epoch in range(10001):
        # perform forward pass on training data
        dense1.forward(X)

        # perform forward pass through relu_activation function
        relu_activation.forward(dense1.output)

        # perform forward pass through 2nd layer
        dense2.forward(relu_activation.output)

        # perform forward pass through activation/loss function
        # softmax activation
        # categorical crossentropy loss
        loss = loss_activation.forward(dense2.output, y)

        """
        #break out softmax and loss function
        softmax_activation = Activation_Softmax()
        softmax_activation.forward(dense2.output)

        loss_function = Loss_CategoricalCrossEntropy()
        loss2 = loss_function.calculate(softmax_activation.output, y)
        """

        # calculate accuracy from output
        predictions = np.argmax(
            loss_activation.output, axis=1
        )  # selects output with highest prob

        # if one-hot-encoded, change to a list of y_true values
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(
                f"epoch : {epoch}, "
                + f"acc: {accuracy:.3f}, "
                + f"loss: {loss:.3f}, "
                + f"lr: {optimizer.current_learning_rate}"
            )

        # backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        relu_activation.backward(dense2.dinputs)
        dense1.backward(relu_activation.dinputs)

        # update network layer's parameters
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
