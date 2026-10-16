"""Five cheap signals that might tell a learner where experience is worth getting.

Each one looks at a batch of fresh pushes, meaning pushes the network has not trained
on yet, and returns one number per push. A higher number means "more to learn here".
Raw error is the floor: it is kept to show how bad a signal can be.

Step 1 asks which of these can be trusted. The judge is in answer_key.py.
"""

import numpy as np


def raw_error(predictions, results):
    """How big the miss was. This is plain novelty: go wherever you were most wrong."""
    return ((predictions - results) ** 2).sum(axis=1)


def repeat_agreement(predictions, first_results, second_results):
    """Do the same push twice and multiply the two misses.

    A miss has a part that comes back every time and a part that is chance.
    Multiplying two misses on the same push keeps the part that comes back,
    while the chance parts cancel out on average.
    """
    first_miss = predictions - first_results
    second_miss = predictions - second_results
    return (first_miss * second_miss).sum(axis=1)


def slope_agreement(network, inputs, results):
    """Does this push ask for the same weight changes as the other pushes do?

    A push's slope is the change it asks for in every weight. The score for a push
    is its slope multiplied with the average slope of the other pushes in the batch.

    For one layer, a push's slope is (what went into the layer) x (the error signal
    that came back to the layer). So two pushes' slopes can be multiplied layer by
    layer, without ever writing a slope out in full.
    """
    _, layer_inputs, layer_errors = network.backward(inputs, results)
    count = len(inputs)

    # agreement[i, j] = slope of push i multiplied with slope of push j
    agreement = np.zeros((count, count))
    for layer_input, layer_error in zip(layer_inputs, layer_errors):
        inputs_alike = layer_input @ layer_input.T + 1  # the +1 covers the bias weights
        errors_alike = layer_error @ layer_error.T
        agreement += inputs_alike * errors_alike

    np.fill_diagonal(agreement, 0)  # a push does not get to agree with itself
    return agreement.sum(axis=1) / (count - 1)


def progress_over_time(past_network, network, inputs, results):
    """How much better is the network now than its past self, on pushes neither has trained on?

    If the network got better at a fresh push, it must have been learning from
    pushes like it. Chance cannot be learned, so on the toy the two errors stay
    the same and the difference is zero.
    """
    past_error = raw_error(past_network.predict(inputs), results)
    error_now = raw_error(network.predict(inputs), results)
    return past_error - error_now


def forward_progress(predictions_before, predictions_after, results):
    """How much easier did these pushes get during one round of training?

    The pushes themselves were held out of that training, so their own results cannot
    have been memorized. Only learning from other pushes can make them easier to predict.
    The chance part of a result is the same number before and after, so on its own it
    cannot make a difference.
    """
    return raw_error(predictions_before, results) - raw_error(predictions_after, results)
