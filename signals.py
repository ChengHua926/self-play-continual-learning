"""Cheap signals that might tell a learner where experience is worth getting.

Each one looks at a batch of fresh pushes, meaning pushes the network has not trained
on yet, and returns one number per push. A higher number means "more to learn here".
Raw error is the floor: it is kept to show how bad a signal can be.

Step 1 asks which of the first five can be trusted. The judge is in answer_key.py.
Step B adds variants of slope agreement (at the bottom) and screens them all.
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
    """
    agreement = slope_agreements(network, inputs, results)
    np.fill_diagonal(agreement, 0)  # a push does not get to agree with itself
    return agreement.sum(axis=1) / (len(inputs) - 1)


def slope_agreements(network, inputs, results):
    """agreement[i, j]: the slope of push i multiplied with the slope of push j, for every pair.

    For one layer, a push's slope is (what went into the layer) x (the error signal
    that came back to the layer). So two pushes' slopes can be multiplied layer by
    layer, without ever writing a slope out in full.
    """
    _, layer_inputs, layer_errors = network.backward(inputs, results)
    count = len(inputs)
    agreement = np.zeros((count, count))
    for layer_input, layer_error in zip(layer_inputs, layer_errors):
        inputs_alike = layer_input @ layer_input.T + 1  # the +1 covers the bias weights
        errors_alike = layer_error @ layer_error.T
        agreement += inputs_alike * errors_alike
    return agreement


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


# Variants of slope agreement, added in step B.

SIMILAR = 8  # how many similar pushes the variants compare a push with


def similar_pushes(network, inputs):
    """For every push, the SIMILAR other pushes nearest to it in the network's own features.

    No object labels are needed. The hope was that pushes on the same object, or with the
    same effect, lie near each other in the features. Checked before step B ran: in worlds
    of 24 objects only 10-17% of the nearest pushes are on the same object, so the variants
    below still compare a push mostly with pushes on other objects.
    """
    features = network.features(inputs)
    squared_distance = ((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(squared_distance, np.inf)  # a push is not similar to itself
    return np.argsort(squared_distance, axis=1)[:, :SIMILAR]


def slope_agreement_with_similar(agreement, similar):
    """Variant (a): agree with the most similar pushes only, not with all of them.

    agreement comes from slope_agreements and similar from similar_pushes. It is meant to
    mend the flaw of slope agreement, which also compares a push with pushes on other
    objects; see similar_pushes for how far it does.
    """
    return np.take_along_axis(agreement, similar, axis=1).mean(axis=1)


def slope_agreement_repeated(network, inputs, first_results, second_results):
    """Variant (b): the slope from a push's first result multiplied with the slope from its second.

    The chance parts of the two results ask for unrelated changes, which cancel on average.
    The part that comes back every time asks for the same change twice. With the last
    layer's slopes only, this is repeat agreement times 4 (1 + the push's features, squared
    and summed): a multiple that differs from push to push.
    """
    _, layer_inputs, first_errors = network.backward(inputs, first_results)
    _, _, second_errors = network.backward(inputs, second_results)
    return sum(
        ((layer_input**2).sum(axis=1) + 1) * (first * second).sum(axis=1)  # the +1 covers the bias weights
        for layer_input, first, second in zip(layer_inputs, first_errors, second_errors)
    )


def last_layer_agreement_with_similar(network, inputs, results, similar):
    """Variant (c): variant (a) with the last layer's slopes only.

    These are cheap at any size, since a push's last-layer slope is just its features times
    its error.
    """
    features = network.features(inputs)
    error = 2 * (network.predict(inputs) - results)
    agreement = (features @ features.T + 1) * (error @ error.T)
    return np.take_along_axis(agreement, similar, axis=1).mean(axis=1)


def slope_consistency(agreement, similar):
    """Do the slopes of a push and its most similar pushes add up, or cancel out?

    The size of their summed slope, squared, divided by the sum of their own sizes, squared.
    It is SIMILAR + 1 when all of them point the same way with the same size, and about 1
    when they point in unrelated directions, as noise does. It stays the same if all the
    slopes grow by the same factor, but one slope much bigger than the rest pulls it towards 1.
    """
    group = np.column_stack([np.arange(len(agreement)), similar])  # each push with its similar pushes
    together = agreement[group[:, :, None], group[:, None, :]].sum(axis=(1, 2))
    apart = agreement[group, group].sum(axis=1)
    return together / apart
