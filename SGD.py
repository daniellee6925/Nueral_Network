import numpy as np


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
