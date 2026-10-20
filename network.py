"""A small plain neural network, written out by hand so every step is visible.

Forward pass:  12 input numbers go through three hidden layers and come out as 3
               predicted numbers.
Backward pass: the error is passed back through the layers. That tells us, for every
               weight, which way to nudge it to make the error smaller. That direction
               is the weight's slope.
Training step: nudge every weight a little, against its slope.

With predicts_noise, the network also guesses how scattered each result is (its variance),
and it is trained on a loss that makes a miss cost less where it expects scatter. Then it
has no reason to chase a toy's random results: it can say "this one is noisy" instead.
"""

import copy

import numpy as np

LOG_VARIANCE_RANGE = (-6.0, 3.0)  # the guessed log-variance is kept inside this range


class Network:
    def __init__(self, rng, sizes=(12, 128, 128, 128, 3), learning_rate=1e-3, predicts_noise=False):
        # With predicts_noise, the last layer gives two numbers per result: the guess, and the
        # log of how scattered the network expects that result to be.
        self.result_size = sizes[-1]
        self.predicts_noise = predicts_noise
        if predicts_noise:
            sizes = (*sizes[:-1], 2 * sizes[-1])

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
        """The forward pass on its own: the network's guess for each result."""
        _, outputs = self._forward(inputs)
        return outputs[:, :self.result_size]

    def predicted_noise(self, inputs):
        """How scattered the network expects each result to be (its variance). Needs predicts_noise."""
        _, outputs = self._forward(inputs)
        log_variance, _ = self._log_variance(outputs)
        return np.exp(log_variance)

    def features(self, inputs):
        """What the last hidden layer makes of the inputs. The critic in step 2 reads these."""
        layer_inputs, _ = self._forward(inputs)
        return layer_inputs[-1]

    def backward(self, inputs, targets, noise=None):
        """Forward pass, then pass the error back through the layers.

        Returns the guesses and, for every layer, two things per example:

            layer_inputs[k]   what went into layer k
            layer_errors[k]   how much the loss changes when layer k's output changes

        Multiply the two and you have that example's slope for layer k's weights.
        Nothing is changed here. This only measures.

        noise, if given, is how scattered each result is expected to be (its variance, one
        number per result). A miss then costs less where more scatter is expected: the loss
        becomes miss² / (2 noise), as for a network that predicts its noise, but with the
        noise taken from elsewhere. Only for a network without predicts_noise.
        """
        layer_inputs, outputs = self._forward(inputs)
        guesses = outputs[:, :self.result_size]

        if self.predicts_noise:
            # The loss for one example, summed over the results: miss² / (2 variance) + log(variance) / 2.
            # Where the network expects much scatter, a miss costs little, so it stops chasing noise.
            log_variance, inside = self._log_variance(outputs)
            variance = np.exp(log_variance)
            miss = guesses - targets
            error = np.concatenate([miss / variance, (0.5 - miss**2 / (2 * variance)) * inside], axis=1)
        elif noise is not None:
            error = (guesses - targets) / noise
        else:
            # The loss for one example is its squared miss, summed over the 3 outputs.
            error = 2 * (guesses - targets)
        return guesses, layer_inputs, self._pass_back(layer_inputs, error)

    def output_slopes(self, inputs, output):
        """How one output (not the loss) moves with every weight, per example.

        Returned the way backward returns them: what went into each layer, and the error
        signal that came back to it, with the error set to 1 at the chosen output and 0 at
        the others.
        """
        layer_inputs, outputs = self._forward(inputs)
        error = np.zeros_like(outputs)
        error[:, output] = 1
        return layer_inputs, self._pass_back(layer_inputs, error)

    def train_step(self, inputs, targets, noise=None):
        """Learn a little from one batch of examples. For noise, see backward."""
        _, layer_inputs, layer_errors = self.backward(inputs, targets, noise)
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
        outputs = layer_inputs[-1] @ self.weights[-1] + self.biases[-1]
        return layer_inputs, outputs

    def _pass_back(self, layer_inputs, error):
        """Walk the error back from the last layer to the first. Returns the error at every layer."""
        layer_errors = [error]
        for k in range(len(self.weights) - 1, 0, -1):
            was_switched_on = layer_inputs[k] > 0
            error = (error @ self.weights[k].T) * was_switched_on
            layer_errors.insert(0, error)
        return layer_errors

    def _log_variance(self, outputs):
        """The guessed log-variance, kept inside LOG_VARIANCE_RANGE, and yes/no per number: was it inside?"""
        raw = outputs[:, self.result_size:]
        low, high = LOG_VARIANCE_RANGE
        return np.clip(raw, low, high), (raw > low) & (raw < high)

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
