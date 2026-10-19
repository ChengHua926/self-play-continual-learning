"""Step 2: let the learner choose its own pushes. Does it learn the world faster?

Each learner is the step-1 network plus a critic (critic.py): a plain head that reads
the network's last hidden layer and forecasts how good a push would be. Every round:

    1. draw 1024 random candidate pushes, and let the critic score them
    2. do the 96 best ones plus 32 random ones, so that every object keeps being pushed.
       Do each push twice.
    3. split the pushes into halves A and B, as in step 1
    4. train on memory, which now holds A but not B
    5. grade B with the learner's label, and let the critic learn from the grades
    6. put B into memory
    7. measure the true error of each object, against the world's rules

The learners differ only in the label their critic learns:

    random pushing       no critic: all 128 pushes are random, as in step 1 (wood gets 40%)
    even pushing         no critic: all 128 pushes are random, spread evenly over the objects.
                         A fairer baseline: part of any gain may come from just not
                         pushing wood so often.
    raw error            the floor
    repeat agreement     error that comes back when a push is repeated
    forward progress     how much easier the held-out pushes got during the round

Every learner runs with two training settings: 100 steps per round, where the network
memorizes the toy's random results, and 20 steps per round, where it memorizes much less.

The map (run_map.py) uses the same run function in bigger and narrow worlds, adds the two
other labels, progress over time and slope agreement, and can switch on a guard against
memorizing noise: "early stopping" (see train below) or "predicts its noise" (a network
that also guesses how scattered each result is, see network.py).

Run it with:   python3 run_step2.py
It takes about twenty seconds, writes results/step2.npz, and prints the report.
"""

from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import report_step2
from answer_key import true_error_per_object
from critic import Critic
from network import Network
from run_step1 import BATCH_SIZE, PAST_SELF_AGE, PUSHES_PER_ROUND, ROUNDS, SEEDS, SHARE, TEST_PUSHES
from signals import forward_progress, progress_over_time, raw_error, repeat_agreement, slope_agreement
from world import RESULT_SIZE, TABLE, joined

BASELINES = ("random pushing", "even pushing")  # learners without a critic
LEARNERS = (*BASELINES, "raw error", "repeat agreement", "forward progress")
TRAINING = (100, 20)     # training steps per round
CANDIDATES = 1024        # random pushes the critic scores every round
LEFT_TO_CHANCE = 32      # pushes per round that stay random, so that every object keeps being pushed

# Guards against memorizing noise, used by the map.
GUARDS = ("none", "early stopping", "predicts its noise")

# Early stopping: each round, a few pushes of half A go into a validation set that is never
# trained on. Training checks the error on that set every few steps, and stops the round
# once the error stops falling.
VALIDATION_PER_ROUND = 8  # pushes per round kept for validation
CHECK_EVERY = 10          # training steps between two checks
PATIENCE = 2              # checks in a row without improvement before the round's training stops


def run(learner, train_steps, seed, world=TABLE, share=SHARE, guard="none"):
    """One learner, one training setting, one seed, in one world.

    share says how often each object is picked when pushes are left to chance. guard is
    one of GUARDS.
    Returns, for every round and object, the true error and how many pushes went to that
    object, and for every round the number of training steps actually taken.
    """
    rng = np.random.default_rng(seed)
    network = Network(
        rng, sizes=(world.input_size, 128, 128, 128, RESULT_SIZE), predicts_noise=(guard == "predicts its noise"),
    )
    critic = Critic()
    memory_inputs = np.zeros((0, world.input_size))
    memory_results = np.zeros((0, RESULT_SIZE))
    validation_inputs = np.zeros((0, world.input_size))
    validation_results = np.zeros((0, RESULT_SIZE))
    past_selves = []
    test_pushes = world.test_pushes(TEST_PUSHES, rng)

    true_error = np.full((ROUNDS, len(world.names)), np.nan)
    pushes_made = np.zeros((ROUNDS, len(world.names)))
    steps_taken = np.zeros(ROUNDS)

    for round_number in range(ROUNDS):
        on_table = world.appears_at <= round_number

        # Choose the pushes, and do every push twice.
        if learner == "random pushing":
            pushes = world.random_pushes(PUSHES_PER_ROUND, share * on_table, rng)
        elif learner == "even pushing":
            pushes = world.random_pushes(PUSHES_PER_ROUND, on_table, rng)
        else:
            pushes = choose_pushes(critic, network, world, on_table, share, rng)
        pushes_made[round_number] = np.bincount(pushes.obj, minlength=len(world.names))
        inputs = world.as_inputs(pushes)
        first_results = world.push(pushes, rng)
        second_results = world.push(pushes, rng)

        # Half A is trained on this round. Half B is held out until the next round, as in step 1.
        half_b = rng.permutation(PUSHES_PER_ROUND) < PUSHES_PER_ROUND // 2
        half_a = ~half_b
        inputs_b, first_b, second_b = inputs[half_b], first_results[half_b], second_results[half_b]
        predictions_before = network.predict(inputs_b)

        # Every label but forward progress is read now, before any training.
        grades = grade_before_training(learner, network, past_selves, predictions_before, inputs_b, first_b, second_b)

        # Keep a snapshot for "progress over time", as in step 1.
        if round_number > 0:
            past_selves = (past_selves + [network.copy()])[-PAST_SELF_AGE:]

        # With early stopping, a few random pushes of A go into the validation set instead of memory.
        early_stopping = guard == "early stopping"
        if early_stopping:
            kept = rng.choice(np.flatnonzero(half_a), VALIDATION_PER_ROUND, replace=False)
            half_a[kept] = False
            validation_inputs = np.concatenate([validation_inputs, inputs[kept], inputs[kept]])
            validation_results = np.concatenate([validation_results, first_results[kept], second_results[kept]])

        memory_inputs = np.concatenate([memory_inputs, inputs[half_a], inputs[half_a]])
        memory_results = np.concatenate([memory_results, first_results[half_a], second_results[half_a]])
        validation = (validation_inputs, validation_results) if early_stopping else None
        steps_taken[round_number] = train(network, memory_inputs, memory_results, train_steps, rng, validation)

        # Forward progress needs the network after training. Then the critic learns from the grades.
        if learner == "forward progress":
            grades = forward_progress(predictions_before, network.predict(inputs_b), first_b)
        if grades is not None:
            critic.learn(network, inputs_b, grades)

        # B goes into memory now, so that it is trained on from the next round on.
        memory_inputs = np.concatenate([memory_inputs, inputs_b, inputs_b])
        memory_results = np.concatenate([memory_results, first_b, second_b])

        true_error[round_number] = true_error_per_object(network, test_pushes, on_table, world)

    return true_error, pushes_made, steps_taken


def train(network, memory_inputs, memory_results, train_steps, rng, validation=None):
    """Train for up to train_steps steps on batches drawn from memory. Returns the steps taken.

    validation, if given, is (inputs, results) of pushes that are never trained on. Their
    error is checked every CHECK_EVERY steps, and training stops once it has failed to
    improve PATIENCE checks in a row. This is the standard guard against memorizing,
    called early stopping.
    """
    if validation is not None:
        best, misses = validation_error(network, *validation), 0
    for step in range(1, train_steps + 1):
        replayed = rng.integers(0, len(memory_inputs), BATCH_SIZE)
        network.train_step(memory_inputs[replayed], memory_results[replayed])
        if validation is not None and step % CHECK_EVERY == 0:
            error = validation_error(network, *validation)
            if error < best:
                best, misses = error, 0
            else:
                misses += 1
                if misses == PATIENCE:
                    return step
    return train_steps


def validation_error(network, inputs, results):
    """The average squared miss on pushes the network has never trained on."""
    return raw_error(network.predict(inputs), results).mean()


def choose_pushes(critic, network, world, on_table, share, rng):
    """The critic's favourite candidates, plus a few pushes left to chance."""
    # Candidates come from every object on the table equally often, so the critic alone decides the mix.
    candidates = world.random_pushes(CANDIDATES, on_table, rng)
    best_first = np.argsort(-critic.score(network, world.as_inputs(candidates)))
    chosen = candidates.take(best_first[:PUSHES_PER_ROUND - LEFT_TO_CHANCE])
    left_to_chance = world.random_pushes(LEFT_TO_CHANCE, share * on_table, rng)
    return joined(chosen, left_to_chance)


def grade_before_training(learner, network, past_selves, predictions_before, inputs, first_results, second_results):
    """This learner's grades for the held-out pushes, if its label can be read before training.

    Returns nothing (None) for the baselines and for forward progress, which needs the
    network after training. Progress over time also grades nothing in the first rounds,
    when there is no past self to compare with yet.
    """
    if learner == "raw error":
        return raw_error(predictions_before, first_results)
    if learner == "repeat agreement":
        return repeat_agreement(predictions_before, first_results, second_results)
    if learner == "slope agreement":
        return slope_agreement(network, inputs, first_results)
    if learner == "progress over time" and len(past_selves) == PAST_SELF_AGE:
        return progress_over_time(past_selves[0], network, inputs, first_results)
    return None


if __name__ == "__main__":
    jobs = list(product(LEARNERS, TRAINING, SEEDS))
    with Pool() as pool:  # the runs do not depend on each other, so do them side by side
        results = pool.starmap(run, jobs)

    # Arrays are [learner, training setting, run, round, object].
    shape = (len(LEARNERS), len(TRAINING), len(SEEDS), ROUNDS, len(TABLE.names))
    Path("results").mkdir(exist_ok=True)
    np.savez(
        report_step2.RESULTS_FILE,
        true_error=np.array([error for error, _, _ in results]).reshape(shape),
        pushes_made=np.array([made for _, made, _ in results]).reshape(shape),
        learners=LEARNERS, training=TRAINING, objects=TABLE.names, appears_at=TABLE.appears_at, share=SHARE,
    )
    report_step2.main()
