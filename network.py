"""A small plain neural network, written out by hand so every step is visible.

Forward pass:  12 input numbers go through three hidden layers and come out as 3
               predicted numbers.
Backward pass: the error is passed back through the layers. That tells us, for every
               weight, which way to nudge it to make the error smaller. That direction
               is the weight's slope.
Training step: nudge every weight a little, against its slope.
"""

import copy

import numpy as np


class Network:
    def __init__(self, rng, sizes=(12, 128, 128, 128, 3), learning_rate=1e-3):
        # One table of weights and one list of biases per layer.
        self.weights = [
            rng.normal(0, np.sqrt(2 / size_in), (size_in, size_out))
            for size_in, size_out in zip(sizes[:-1], sizes[1:])
        ]
        self.biases = [np.zeros(size_out) for size_out in sizes[1:]]
        self.learning_rate = learning_rate

        # Memory for the optimizer (see _nudge).
        self._average_slope = [np.zeros_like(p) for p in self._parameters()]
        self._average_squared_slope = [np.zeros_like(p) for p in self._parameters()]
        self._steps_taken = 0

    def predict(self, inputs):
        """The forward pass on its own."""
        _, predictions = self._forward(inputs)
        return predictions

    def features(self, inputs):
        """What the last hidden layer makes of the inputs. The critic in step 2 reads these."""
        layer_inputs, _ = self._forward(inputs)
        return layer_inputs[-1]

    def backward(self, inputs, targets):
        """Forward pass, then pass the error back through the layers.

        Returns the predictions and, for every layer, two things per example:

            layer_inputs[k]   what went into layer k
            layer_errors[k]   how much the loss changes when layer k's output changes

        Multiply the two and you have that example's slope for layer k's weights.
        Nothing is changed here. This only measures.
        """
        layer_inputs, predictions = self._forward(inputs)

        # The loss for one example is its squared miss, summed over the 3 outputs.
        error = 2 * (predictions - targets)
        layer_errors = [error]

        # Walk back from the last layer to the first.
        for k in range(len(self.weights) - 1, 0, -1):
            was_switched_on = layer_inputs[k] > 0
            error = (error @ self.weights[k].T) * was_switched_on
            layer_errors.insert(0, error)

        return predictions, layer_inputs, layer_errors

    def train_step(self, inputs, targets):
        """Learn a little from one batch of examples."""
        _, layer_inputs, layer_errors = self.backward(inputs, targets)
        count = len(inputs)

        # The batch's slope is the average of its examples' slopes.
        weight_slopes = [
            layer_input.T @ layer_error / count
            for layer_input, layer_error in zip(layer_inputs, layer_errors)
        ]
        bias_slopes = [layer_error.mean(axis=0) for layer_error in layer_errors]
        self._nudge(weight_slopes + bias_slopes)

    def copy(self):
        """A frozen snapshot of the network as it is right now."""
        return copy.deepcopy(self)

    def _forward(self, inputs):
        layer_inputs = [inputs]
        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            switched_on = np.maximum(layer_inputs[-1] @ weights + biases, 0)
            layer_inputs.append(switched_on)
        predictions = layer_inputs[-1] @ self.weights[-1] + self.biases[-1]
        return layer_inputs, predictions

    def _parameters(self):
        return self.weights + self.biases

    def _nudge(self, slopes):
        """Move every weight against its slope, using the Adam rule.

        Adam keeps a running average of each weight's slope and of its squared slope.
        Dividing one by the other gives every weight a step of a sensible size,
        whether its slopes are usually large or usually small.
        """
        self._steps_taken += 1
        for parameter, slope, average, average_squared in zip(
            self._parameters(), slopes, self._average_slope, self._average_squared_slope
        ):
            average[:] = 0.9 * average + 0.1 * slope
            average_squared[:] = 0.999 * average_squared + 0.001 * slope**2

            # Both averages start at zero, which makes them too small early on. Correct for that.
            corrected_average = average / (1 - 0.9**self._steps_taken)
            corrected_squared = average_squared / (1 - 0.999**self._steps_taken)

            parameter -= self.learning_rate * corrected_average / (np.sqrt(corrected_squared) + 1e-8)
