"""Measures read before a push is made: might there be something to learn here?

The labels in signals.py look at a push's result. These never do. They look only at the
learner itself (its weights, its features, its past) and at the pushes it has seen. So a
learner could choose by them directly, with no critic. The flip side: they cannot know that
a push's result would surprise the learner, only that the push is unlike what it knows.

    pinned weights          does the prediction rest on weights the data has barely pinned down?
    prediction drift        has the learner's guess for this push changed in the last rounds?
    feature novelty         how far is this push from every push seen, in the learner's own features?
    ensemble disagreement   do five learners, trained on the same memory, disagree here?
    random-network novelty  how badly does a small network, trained to imitate a fixed random
                            network on every push seen, imitate it here?
"""

import numpy as np

FISHER_SAMPLE = 1024  # records of memory used to measure how pinned each weight is
NEAREST = 5           # feature novelty is the average distance to this many nearest pushes seen


def pinned_weight_uncertainty(network, inputs, memory_inputs, memory_results, rng):
    """How loosely the data so far pins down the weights that each push's prediction rests on.

    Before any data, every weight is as loose as its random start: its variance is the
    variance it was drawn with. Every record in memory that moves a prediction through a
    weight pins that weight down a bit more: in proportion to the squared slope of the
    prediction with respect to that weight, and divided by how big the network's misses
    are on memory (the noisier the records, the less each one pins). This is the Laplace
    approximation, with only the diagonal of the Gauss-Newton matrix.

    A push's uncertainty then adds up, over all weights, its own squared slope times how
    loose the weight is. A push on an object never seen rests on weights nothing has pinned.
    The pinning is measured on a sample of memory and scaled up to all of memory. Biases,
    which start at zero, are given the same starting looseness as their layer's weights.
    """
    picked = rng.choice(len(memory_inputs), min(FISHER_SAMPLE, len(memory_inputs)), replace=False)
    sample = memory_inputs[picked]
    noise = ((network.predict(sample) - memory_results[picked]) ** 2).mean()  # the average squared miss

    # How much the records pin each weight: the sum over records and outputs of the squared slope.
    weight_pinning = [np.zeros_like(weights) for weights in network.weights]
    bias_pinning = [np.zeros_like(biases) for biases in network.biases]
    for output in range(network.result_size):
        layer_inputs, layer_errors = network.output_slopes(sample, output)
        for k, (layer_input, layer_error) in enumerate(zip(layer_inputs, layer_errors)):
            weight_pinning[k] += (layer_input**2).T @ layer_error**2
            bias_pinning[k] += (layer_error**2).sum(axis=0)
    scale = len(memory_inputs) / len(sample) / noise
    prior = [weights.shape[0] / 2 for weights in network.weights]  # 1 / the variance each layer was drawn with
    weight_looseness = [1 / (scale * pinning + start) for pinning, start in zip(weight_pinning, prior)]
    bias_looseness = [1 / (scale * pinning + start) for pinning, start in zip(bias_pinning, prior)]

    # Each push: its squared slopes, weighted by how loose each weight is.
    uncertainty = np.zeros(len(inputs))
    for output in range(network.result_size):
        layer_inputs, layer_errors = network.output_slopes(inputs, output)
        for k, (layer_input, layer_error) in enumerate(zip(layer_inputs, layer_errors)):
            uncertainty += ((layer_input**2 @ weight_looseness[k]) * layer_error**2).sum(axis=1)
            uncertainty += layer_error**2 @ bias_looseness[k]
    return uncertainty


def prediction_drift(past_network, network, inputs):
    """How much the learner's guess for each push changed since its past self."""
    return ((network.predict(inputs) - past_network.predict(inputs)) ** 2).sum(axis=1)


def feature_novelty(network, inputs, seen_inputs):
    """How far each push is from the pushes seen so far, in the network's last hidden layer.

    The average distance to the NEAREST closest pushes seen.
    """
    features, seen = network.features(inputs), network.features(seen_inputs)
    squared = (features**2).sum(axis=1)[:, None] + (seen**2).sum(axis=1)[None, :] - 2 * features @ seen.T
    nearest = np.partition(np.maximum(squared, 0), NEAREST - 1, axis=1)[:, :NEAREST]
    return np.sqrt(nearest).mean(axis=1)


def ensemble_disagreement(members, inputs):
    """How much several networks, trained on the same memory from different starts, disagree."""
    guesses = np.stack([member.predict(inputs) for member in members])
    return guesses.var(axis=0).sum(axis=1)


def random_network_novelty(target, imitator, inputs):
    """How badly the imitator, trained on the pushes seen so far, matches the fixed random target here."""
    return ((imitator.predict(inputs) - target.predict(inputs)) ** 2).sum(axis=1)
