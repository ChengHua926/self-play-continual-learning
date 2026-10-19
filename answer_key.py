"""The answer key: where does extra experience really help?

The world is small, so we can find out by brute force. Make two copies of the learner.
The memory of one copy also holds extra pushes of a single object, treated exactly like
the learner's own new pushes: each push is done twice, both results go into memory, and
training draws its batches from memory. Then train both copies, round after round, the
way the learner trains. From the second round on, both copies get the same new random
pushes. After every round, compare their true errors.

Whichever object's extra pushes shrink the true error most is where a curious learner
should have gone. The answer can change with how far ahead you look, so the gain is
recorded after 1 round, after 2 rounds, and so on.

This uses the world's rules to score the copies, which a real learner could not do.
It is only here to judge the signals.
"""

import numpy as np

from world import OBJECTS, TABLE, as_inputs, push, random_pushes


def true_error(network, test_pushes, on_table):
    """The network's error against the best possible guess, counting every object on the table equally."""
    return np.nanmean(true_error_per_object(network, test_pushes, on_table))


def true_error_per_object(network, test_pushes, on_table, world=TABLE):
    """The same error, one number per object. Objects not on the table get a gap (not-a-number)."""
    miss = network.predict(world.as_inputs(test_pushes)) - world.expected_result(test_pushes)
    error = (miss**2).sum(axis=1)
    return np.array([
        error[test_pushes.obj == obj].mean() if on_table[obj] else np.nan
        for obj in range(len(world.names))
    ])


def value_of_extra_pushes(
    network, memory_inputs, memory_results, obj, test_pushes, on_table, shares, seed,
    rounds, extra_pushes, pushes_per_round, train_steps, batch_size,
):
    """How much lower is the true error if the learner gets extra pushes of one object now?

    Returns one number per round looked ahead: the gain after 1 round, after 2 rounds, and so on.
    shares says how often each object is picked for the new random pushes of the later rounds.
    """
    rng = np.random.default_rng(seed)

    only_this_object = np.eye(len(OBJECTS))[obj]
    extra = random_pushes(extra_pushes, only_this_object, rng)
    extra_inputs, extra_results = done_twice(extra, rng)

    # What both copies have in memory. The extra pushes are kept apart, for one copy only.
    shared_inputs, shared_results = memory_inputs, memory_results
    without_extra = network.copy()
    with_extra = network.copy()

    gains = []
    for round_number in range(rounds):
        if round_number > 0:
            # Both copies get the same new random pushes, as the learner would.
            new_inputs, new_results = done_twice(random_pushes(pushes_per_round, shares, rng), rng)
            shared_inputs = np.concatenate([shared_inputs, new_inputs])
            shared_results = np.concatenate([shared_results, new_results])

        for _ in range(train_steps):
            # Draw one batch as if the extra pushes were in memory. The copy without them trains on
            # the part of the batch that came from the shared memory. So the extra pushes that land
            # in the batch are the only difference between the two copies.
            batch = rng.integers(0, len(shared_inputs) + len(extra_inputs), batch_size)
            from_shared = batch[batch < len(shared_inputs)]
            from_extra = batch[batch >= len(shared_inputs)] - len(shared_inputs)

            without_extra.train_step(shared_inputs[from_shared], shared_results[from_shared])
            with_extra.train_step(
                np.concatenate([shared_inputs[from_shared], extra_inputs[from_extra]]),
                np.concatenate([shared_results[from_shared], extra_results[from_extra]]),
            )

        gains.append(true_error(without_extra, test_pushes, on_table) - true_error(with_extra, test_pushes, on_table))
    return gains


def done_twice(pushes, rng):
    """Do every push twice, like the learner does, and return what goes into memory."""
    inputs = as_inputs(pushes)
    return np.concatenate([inputs, inputs]), np.concatenate([push(pushes, rng), push(pushes, rng)])
