import numpy as np
import nnfs
import RMSprop
import Adagrad
import SGD
import Adam
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(
        self,
        n_inputs,
        n_nuerons,
        weight_regularizer_L1=0,
        weight_regularizer_L2=0,
        bias_regularizer_L1=0,
        bias_regularizer_L2=0,
    ):
        # set initial weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))
        # set regularization strenght
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

    # forward pass
    def forward(self, inputs):
        # remember input values
        self.inputs = inputs
        # calculate output values
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass
    def backward(self, dvalues):
        # gradients on paremeters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # gradients on regularization
        # L1 on weights
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1

        # L2 on weights
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights

        # L1 on bias
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1

        # L2 on weights
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases

        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dropout:
    def __init__(self, rate) -> None:
        # invert it: dropout of 0.1 -> success rate of 0.9
        self.rate = 1 - rate

    # forward pass
    def forward(self, inputs):
        # save inputs
        self.inputs = inputs
        # generate and save scaled mask
        self.binary_mask = (
            np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        )

        # apply mask to output values
        self.output = inputs * self.binary_mask


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
    # regularization loss calculation
    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights
        # calculate only when factor greater than 0
        if layer.weight_regularizer_L1 > 0:
            regularization_loss += layer.weight_regularizer_L1 * np.sum(
                np.abs(layer.weights)
            )

        # L2 regularization - weights
        if layer.weight_regularizer_L2 > 0:
            regularization_loss += layer.weight_regularizer_L2 * np.sum(
                layer.weights * layer.weights
            )

        # L1 regularization - biases
        if layer.bias_regularizer_L1 > 0:
            regularization_loss += layer.bias_regularizer_L1 * np.sum(
                np.abs(layer.biases)
            )

        # L2 regularization - weights
        if layer.bias_regularizer_L2 > 0:
            regularization_loss += layer.bias_regularizer_L2 * np.sum(
                layer.biases * layer.biases
            )

        return regularization_loss

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


if __name__ == "__main__":
    # create data set
    X, y = spiral_data(100, 3)

    # create first dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 64, weight_regularizer_L2=5e-4, bias_regularizer_L2=5e-4)

    # create RELU activation
    relu_activation = Activation_ReLU()

    # create 2nd dense layer with 3 input features (same as output num. from first layer) and 3 output values
    dense2 = Layer_Dense(64, 3)

    # create softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_loss_CategoricalCrossentropy()

    # create optimizer object
    # optimizer = SGD.Optimizer_SGD(decay=1e-3, momentum=0.9)
    # optimizer = Adagrad.Optimizer_Adagrad(decay=1e-4)
    # optimizer = RMSprop.Optimizer_RMSprop(decay=1e-4)
    optimizer = Adam.Optimizer_Adam(learning_rate=0.02, decay=1e-5)

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
        data_loss = loss_activation.forward(dense2.output, y)

        # calculate regularization penalty
        regularization_loss = loss_activation.loss.regularization_loss(dense1)
        +loss_activation.loss.regularization_loss(dense2)

        # calculate overall loss
        loss = data_loss + regularization_loss
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
                + f"data_loss: {loss:.3f}, "
                + f"reg_loss: {regularization_loss:.3f}, "
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

    # Validate the model

    # create dataset
    X_test, y_test = spiral_data(samples=100, classes=3)

    # perform forward pass on layer and activation
    dense1.forward(X_test)
    relu_activation.forward(dense1.output)
    dense2.forward(relu_activation.output)
    loss = loss_activation.forward(dense2.output, y_test)

    predictions = np.argmax(
        loss_activation.output, axis=1
    )  # selects output with highest prob
    # if one-hot-encoded, change to a list of y_true values
    if len(y.shape) == 2:
        y = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)

    print(f"validation, acc: {accuracy:.3f}, loss: {loss:.3f}")
