"""Step B: screen the candidate signals in the toy worlds, with no critic.

Step A found that the critics barely forecast their labels, and that the way pushes are
chosen has an effect of its own. So step B leaves choosing out. Pushes are random, spread
evenly over the objects on the table, and every signal is only read, never used to choose.
Then each signal is checked against facts built into the worlds (report_step_b.py):

    ignores noise             toys get no more than their share of the signal's top picks
    notices new things        a newly arrived object beats every old object (plain worlds)
    finds rare useful pushes  in narrow worlds, pushes in the narrow parts score higher than
                              other pushes on the same kind of object, and by more than in
                              plain worlds

Every round works as in step 1: 128 random pushes, each done twice so that the signals that
need a repeat can be read. Half A is trained on. Half B is held out, read by every signal,
and put into memory after the round's training. The network trains lightly, 20 steps per
round, so that it memorizes little of the toys' noise; the checks are passed or failed
there. The whole screen is run again at 100 steps per round, where the toys' noise does get
memorized, as a stress test.

Because every push is done twice, as in the old "doubled" budget, nothing here can be
compared with step A's fair runs. Nothing needs to be: no learner chooses anything.

The signals. Labels, read after a push is made (signals.py):

    raw error                 the floor
    repeat agreement
    slope agreement           compared with all other pushes, as before
    slope, similar pushes     compared only with the 8 most similar pushes (variant a)
    slope, same push twice    the slope from the first result times the slope from the second (variant b)
    slope, last layer         variant a, with the last layer's slopes only (variant c)
    slope consistency         do the slopes of a push and its similar pushes add up or cancel?
    progress over time
    progress, new first       the same, but a push on an object seen for fewer than 3 rounds
                              comes first. It passes "notices new things" by construction.
    forward progress

Measures, read before a push is made (measures.py):

    pinned weights            a Laplace approximation: does the prediction rest on loose weights?
    prediction drift          did the guess for this push change in the last 3 rounds?
    feature novelty           how far from every push seen, in the network's own features?
    ensemble disagreement     do 5 networks, trained on the same memory, disagree?
    random-network novelty    how badly does a network imitate a fixed random one here?

And three controls, which know nothing about learning. They show what each check gives
to a signal that has no business passing it:

    predicted size            how big the network expects the result to be
    push strength             how hard the push is
    random number             a random number for every push

Two noise models are read too, for check 4: they guess how scattered each result is.

    noise from repeats        a network trained on pushes done twice, to predict half the
                              squared difference of the two results (run_step2.repeat_noise)
    predicts its noise        a network that predicts its noise (network.py), as in the map

Part 5 runs the noise models as guards, with even pushing and 50 training steps per round,
through the run function of step 2: no guard, "predicts its noise", "noise from repeats".

Run it with:   python3 run_step_b.py
It takes a few minutes, writes results/step_b.npz and results/step_b_guards.npz, and
prints the report.
"""

from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import report_step_b
from measures import (
    ensemble_disagreement, feature_novelty, pinned_weight_uncertainty, prediction_drift, random_network_novelty,
)
from network import Network
from run_map import build_world, run_in_its_world
from run_step1 import PAST_SELF_AGE, PUSHES_PER_ROUND, ROUNDS
from run_step2 import repeat_noise, train
from signals import (
    forward_progress, last_layer_agreement_with_similar, progress_over_time, raw_error, repeat_agreement,
    similar_pushes, slope_agreement, slope_agreement_repeated, slope_agreement_with_similar, slope_agreements,
    slope_consistency,
)
from world import RESULT_SIZE

SIGNALS = (
    "raw error", "repeat agreement", "slope agreement", "slope, similar pushes", "slope, same push twice",
    "slope, last layer", "slope consistency", "progress over time", "progress, new first", "forward progress",
    "pinned weights", "prediction drift", "feature novelty", "ensemble disagreement", "random-network novelty",
    "predicted size", "push strength", "random number",
)
NOISE_MODELS = ("noise from repeats", "predicts its noise")
KINDS = ("box", "roller", "spinner", "toy")
WORLDS = ("6", "24", "narrow 6", "narrow 24")
TRAINING = (20, 100)  # training steps per round: light, where the checks are decided, and heavy, the stress test
SEEDS = tuple(range(10))
ENSEMBLE = 5  # networks in the ensemble, the learner's own network included

# Part 5: the noise models as guards.
GUARDS = ("none", "predicts its noise", "noise from repeats")
BUDGETS = ("doubled", "fair")
GUARD_TRAIN_STEPS = 50


def screen(train_steps, world_name, seed):
    """One run in one world. Every signal and noise model is read on half B, every round.

    Returns their values, and four facts about each push of B (from the world, not known to
    the learner): its object, its object's kind, whether it lands in its object's narrow
    part, and whether its object arrived this round or the round before. All are [round, push].
    """
    rng = np.random.default_rng(seed)
    world = build_world(world_name, seed)
    sizes = (world.input_size, 128, 128, 128, RESULT_SIZE)
    network = Network(rng, sizes)
    others = [Network(rng, sizes) for _ in range(ENSEMBLE - 1)]  # the rest of the ensemble
    target = Network(rng, (world.input_size, 64, 64, 16))        # the fixed random network
    imitator = Network(rng, (world.input_size, 64, 64, 16))      # learns to imitate it on the pushes seen
    noise_from_repeats = Network(rng, sizes)
    predicts_its_noise = Network(rng, sizes, predicts_noise=True)

    memory_inputs = np.zeros((0, world.input_size))
    memory_results = np.zeros((0, RESULT_SIZE))
    pair_inputs = np.zeros((0, world.input_size))   # every push once, with half the squared difference
    pair_noise = np.zeros((0, RESULT_SIZE))         # of its two results (for the noise model)
    past_selves = []
    first_pushed = np.full(len(world.names), -1)    # the round each object was first pushed in

    held_out = PUSHES_PER_ROUND // 2
    values = {name: np.full((ROUNDS, held_out), np.nan) for name in SIGNALS + NOISE_MODELS}
    objects = np.zeros((ROUNDS, held_out), dtype=int)
    kind = np.zeros((ROUNDS, held_out), dtype=int)
    narrow_part = np.zeros((ROUNDS, held_out), dtype=bool)
    new = np.zeros((ROUNDS, held_out), dtype=bool)

    for round_number in range(ROUNDS):
        on_table = world.appears_at <= round_number
        pushes = world.random_pushes(PUSHES_PER_ROUND, on_table, rng)
        inputs = world.as_inputs(pushes)

        # Half A is trained on this round. Half B is held out until after the training, as in step 1.
        half_b = rng.permutation(PUSHES_PER_ROUND) < held_out
        half_a = ~half_b
        inputs_b, objects_b = inputs[half_b], pushes.obj[half_b]

        # What each push of B is, for the checks.
        objects[round_number] = objects_b
        kind[round_number] = [KINDS.index(world.kinds[obj]) for obj in objects_b]
        narrow_part[round_number] = world.in_narrow_part(pushes.take(half_b))
        arrived = world.appears_at[objects_b]
        new[round_number] = (arrived > 0) & (round_number - arrived <= 1)

        # The measures, read before the pushes are made. They need something seen already.
        first_pushed[(first_pushed < 0) & np.isin(np.arange(len(world.names)), pushes.obj)] = round_number
        if round_number > 0:
            values["pinned weights"][round_number] = pinned_weight_uncertainty(
                network, inputs_b, memory_inputs, memory_results, rng)
            values["feature novelty"][round_number] = feature_novelty(network, inputs_b, pair_inputs)
            values["ensemble disagreement"][round_number] = ensemble_disagreement([network, *others], inputs_b)
            values["random-network novelty"][round_number] = random_network_novelty(target, imitator, inputs_b)
            values["noise from repeats"][round_number] = np.maximum(noise_from_repeats.predict(inputs_b), 0).sum(axis=1)
            values["predicts its noise"][round_number] = predicts_its_noise.predicted_noise(inputs_b).sum(axis=1)
        if len(past_selves) == PAST_SELF_AGE:
            values["prediction drift"][round_number] = prediction_drift(past_selves[0], network, inputs_b)
        values["predicted size"][round_number] = (network.predict(inputs_b) ** 2).sum(axis=1)
        values["push strength"][round_number] = pushes.strength[half_b]

        # Make the pushes, each one twice.
        first_results, second_results = world.push(pushes, rng), world.push(pushes, rng)
        first_b, second_b = first_results[half_b], second_results[half_b]

        # The labels, read before any training.
        predictions_before = network.predict(inputs_b)
        agreement = slope_agreements(network, inputs_b, first_b)
        similar = similar_pushes(network, inputs_b)
        values["raw error"][round_number] = raw_error(predictions_before, first_b)
        values["repeat agreement"][round_number] = repeat_agreement(predictions_before, first_b, second_b)
        values["slope agreement"][round_number] = slope_agreement(network, inputs_b, first_b)
        values["slope, similar pushes"][round_number] = slope_agreement_with_similar(agreement, similar)
        values["slope, same push twice"][round_number] = slope_agreement_repeated(network, inputs_b, first_b, second_b)
        values["slope, last layer"][round_number] = last_layer_agreement_with_similar(network, inputs_b, first_b, similar)
        values["slope consistency"][round_number] = slope_consistency(agreement, similar)
        if len(past_selves) == PAST_SELF_AGE:
            progress = progress_over_time(past_selves[0], network, inputs_b, first_b)
            values["progress over time"][round_number] = progress
            seen_for_long = round_number - first_pushed[objects_b] >= PAST_SELF_AGE
            values["progress, new first"][round_number] = np.where(seen_for_long, progress, np.inf)
        values["random number"][round_number] = rng.random(held_out)

        # Keep a snapshot for "progress over time" and "prediction drift", as in step 1.
        if round_number > 0:
            past_selves = (past_selves + [network.copy()])[-PAST_SELF_AGE:]

        # Half A goes into memory, and every network trains on it.
        memory_inputs = np.concatenate([memory_inputs, inputs[half_a], inputs[half_a]])
        memory_results = np.concatenate([memory_results, first_results[half_a], second_results[half_a]])
        pair_inputs = np.concatenate([pair_inputs, inputs[half_a]])
        pair_noise = np.concatenate([pair_noise, repeat_noise(first_results[half_a], second_results[half_a])])
        for member in (network, *others, predicts_its_noise):
            train(member, memory_inputs, memory_results, train_steps, rng)
        train(imitator, pair_inputs, target.predict(pair_inputs), train_steps, rng)
        train(noise_from_repeats, pair_inputs, pair_noise, train_steps, rng)

        # Forward progress needs the network after training.
        values["forward progress"][round_number] = forward_progress(predictions_before, network.predict(inputs_b), first_b)

        # B goes into memory now, so that it is trained on from the next round on.
        memory_inputs = np.concatenate([memory_inputs, inputs_b, inputs_b])
        memory_results = np.concatenate([memory_results, first_b, second_b])
        pair_inputs = np.concatenate([pair_inputs, inputs_b])
        pair_noise = np.concatenate([pair_noise, repeat_noise(first_b, second_b)])

    return values, objects, kind, narrow_part, new


def guarded(budget, world_name, guard, seed):
    """Part 5: even pushing in one world, with one guard and one budget."""
    return run_in_its_world("even pushing", GUARD_TRAIN_STEPS, world_name, guard, seed, budget)


if __name__ == "__main__":
    screens = list(product(TRAINING, WORLDS, SEEDS))
    guard_runs = list(product(BUDGETS, WORLDS, GUARDS, SEEDS))
    with Pool() as pool:  # the runs do not depend on each other, so do them side by side
        results = pool.starmap(screen, screens)
        guard_results = pool.starmap(guarded, guard_runs)
    Path("results").mkdir(exist_ok=True)

    # The screen: every array is saved as [training setting, world, run, round, push].
    def stacked(arrays):
        return np.array(arrays).reshape(len(TRAINING), len(WORLDS), len(SEEDS), *arrays[0].shape)

    values = [result[0] for result in results]
    np.savez_compressed(
        report_step_b.RESULTS_FILE,
        **{name: stacked([run[name].astype(np.float32) for run in values]) for name in SIGNALS + NOISE_MODELS},
        **{fact: stacked([result[f] for result in results])
           for f, fact in enumerate(("objects", "kind", "narrow_part", "new"), start=1)},
        signals=SIGNALS, noise_models=NOISE_MODELS, kinds=KINDS, worlds=WORLDS, training=TRAINING,
    )

    # The guards: every line is saved as [budget, world, guard, run, round].
    shape = (len(BUDGETS), len(WORLDS), len(GUARDS), len(SEEDS), ROUNDS)
    np.savez(
        report_step_b.GUARDS_FILE,
        **{line: np.array([result[line] for result in guard_results]).reshape(shape)
           for line in ("learnable", "toy_error", "steps_taken")},
        budgets=BUDGETS, worlds=WORLDS, guards=GUARDS,
    )
    report_step_b.main()
