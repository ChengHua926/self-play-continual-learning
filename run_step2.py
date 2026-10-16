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

Run it with:   python3 run_step2.py
It takes a few minutes, writes results/step2.npz, and prints the report.
"""

from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import report_step2
from answer_key import true_error_per_object
from critic import Critic
from network import Network
from run_step1 import APPEARS_AT, BATCH_SIZE, PUSHES_PER_ROUND, ROUNDS, SEEDS, SHARE, TEST_PUSHES
from signals import forward_progress, raw_error, repeat_agreement
from world import INPUT_SIZE, OBJECTS, RESULT_SIZE, as_inputs, joined, push, random_pushes

BASELINES = ("random pushing", "even pushing")  # learners without a critic
LEARNERS = (*BASELINES, "raw error", "repeat agreement", "forward progress")
TRAINING = (100, 20)     # training steps per round
CANDIDATES = 1024        # random pushes the critic scores every round
LEFT_TO_CHANCE = 32      # pushes per round that stay random, so that every object keeps being pushed


def run(learner, train_steps, seed):
    """One learner, one training setting, one seed.

    Returns, for every round and object, the true error and how many pushes went to that object.
    """
    rng = np.random.default_rng(seed)
    network = Network(rng)
    critic = Critic()
    memory_inputs = np.zeros((0, INPUT_SIZE))
    memory_results = np.zeros((0, RESULT_SIZE))
    test_pushes = random_pushes(TEST_PUSHES, np.ones(len(OBJECTS)), rng)

    true_error = np.full((ROUNDS, len(OBJECTS)), np.nan)
    pushes_made = np.zeros((ROUNDS, len(OBJECTS)))

    for round_number in range(ROUNDS):
        on_table = APPEARS_AT <= round_number

        # Choose the pushes, and do every push twice.
        if learner == "random pushing":
            pushes = random_pushes(PUSHES_PER_ROUND, SHARE * on_table, rng)
        elif learner == "even pushing":
            pushes = random_pushes(PUSHES_PER_ROUND, on_table, rng)
        else:
            pushes = choose_pushes(critic, network, on_table, rng)
        pushes_made[round_number] = np.bincount(pushes.obj, minlength=len(OBJECTS))
        inputs = as_inputs(pushes)
        first_results = push(pushes, rng)
        second_results = push(pushes, rng)

        # Half A is trained on this round. Half B is held out until the next round, as in step 1.
        half_b = rng.permutation(PUSHES_PER_ROUND) < PUSHES_PER_ROUND // 2
        half_a = ~half_b
        predictions_before = network.predict(inputs[half_b])

        memory_inputs = np.concatenate([memory_inputs, inputs[half_a], inputs[half_a]])
        memory_results = np.concatenate([memory_results, first_results[half_a], second_results[half_a]])
        for _ in range(train_steps):
            replayed = rng.integers(0, len(memory_inputs), BATCH_SIZE)
            network.train_step(memory_inputs[replayed], memory_results[replayed])

        # Grade B with this learner's label, and let the critic learn from the grades.
        if learner not in BASELINES:
            predictions_after = network.predict(inputs[half_b])
            grades = grade(learner, predictions_before, predictions_after, first_results[half_b], second_results[half_b])
            critic.learn(network, inputs[half_b], grades)

        # B goes into memory now, so that it is trained on from the next round on.
        memory_inputs = np.concatenate([memory_inputs, inputs[half_b], inputs[half_b]])
        memory_results = np.concatenate([memory_results, first_results[half_b], second_results[half_b]])

        true_error[round_number] = true_error_per_object(network, test_pushes, on_table)

    return true_error, pushes_made


def choose_pushes(critic, network, on_table, rng):
    """The critic's favourite candidates, plus a few random pushes."""
    # Candidates come from every object on the table equally often, so the critic alone decides the mix.
    candidates = random_pushes(CANDIDATES, on_table, rng)
    best_first = np.argsort(-critic.score(network, as_inputs(candidates)))
    chosen = candidates.take(best_first[:PUSHES_PER_ROUND - LEFT_TO_CHANCE])
    left_to_chance = random_pushes(LEFT_TO_CHANCE, SHARE * on_table, rng)
    return joined(chosen, left_to_chance)


def grade(learner, predictions_before, predictions_after, first_results, second_results):
    """This learner's label, for each held-out push."""
    if learner == "raw error":
        return raw_error(predictions_before, first_results)
    if learner == "repeat agreement":
        return repeat_agreement(predictions_before, first_results, second_results)
    if learner == "forward progress":
        return forward_progress(predictions_before, predictions_after, first_results)
    raise ValueError(f"unknown learner: {learner}")


if __name__ == "__main__":
    jobs = list(product(LEARNERS, TRAINING, SEEDS))
    with Pool() as pool:  # the runs do not depend on each other, so do them side by side
        results = pool.starmap(run, jobs)

    # Arrays are [learner, training setting, run, round, object].
    shape = (len(LEARNERS), len(TRAINING), len(SEEDS), ROUNDS, len(OBJECTS))
    Path("results").mkdir(exist_ok=True)
    np.savez(
        report_step2.RESULTS_FILE,
        true_error=np.array([error for error, _ in results]).reshape(shape),
        pushes_made=np.array([made for _, made in results]).reshape(shape),
        learners=LEARNERS, training=TRAINING, objects=OBJECTS, appears_at=APPEARS_AT, share=SHARE,
    )
    report_step2.main()
