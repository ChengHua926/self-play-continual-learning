"""Step 1: which cheap signal can be trusted to say where experience is worth getting?

There is no curiosity loop yet. The learner pushes at random. Every round:

    1. make 128 new pushes, and do each one twice
    2. split them into two halves: A is trained on this round, B is held out until the next
    3. read the signals on B, before any training on it
    4. work out by brute force where extra pushes would really help, over the next 1 to 10
       rounds (answer_key.py)
    5. train on memory, which now holds A but not B
    6. read B again: forward progress is how much easier B got without being trained on
    7. put B into memory, so that it is trained on from the next round on

report_step1.py then compares the signals with the answer key.

Run it with:   python3 run_step1.py
It takes about fifteen minutes, writes results/step1.npz, and prints the report.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np

import report_step1
from answer_key import value_of_extra_pushes
from network import Network
from signals import forward_progress, progress_over_time, raw_error, repeat_agreement, slope_agreement
from world import INPUT_SIZE, OBJECTS, RESULT_SIZE, as_inputs, push, random_pushes

ROUNDS = 70
PUSHES_PER_ROUND = 128   # new pushes per round. Each one is done twice.
TRAIN_STEPS = 100        # training steps after every round
BATCH_SIZE = 256
PAST_SELF_AGE = 3        # "progress over time" compares with the network from this many rounds ago
SEEDS = (0, 1, 2, 3, 4)  # the whole run is repeated once per seed, and the report averages them

#                      wood  steel  rubber  roller  spinner  toy
SHARE = np.array([     0.40, 0.05,  0.10,   0.15,   0.15,    0.15])  # how often each object is pushed
APPEARS_AT = np.array([0,    20,    0,      0,      40,      0])     # round it is first put on the table

# Raw error comes first: it is the floor, kept to show how bad a signal can be.
SIGNALS = ("raw error", "slope agreement", "repeat agreement", "progress over time", "forward progress")

ANSWER_KEY_REPEATS = 2   # the answer key is worked out this many times per object; the report compares them
ANSWER_KEY_ROUNDS = 10   # how many rounds ahead the answer key looks. This is the slow part.
TEST_PUSHES = 3000       # pushes used to measure the true error


def run(seed):
    """One full run. Returns, for every round and object, the five signals and the answer key.

    A gap (not-a-number) means "not measured": the object was not on the table yet, it had
    fewer than 2 pushes in half B, or it was one of the first rounds, where there is no
    past self to compare with.
    """
    rng = np.random.default_rng(seed)
    network = Network(rng)
    memory_inputs = np.zeros((0, INPUT_SIZE))
    memory_results = np.zeros((0, RESULT_SIZE))
    past_selves = []

    signals = np.full((ROUNDS, len(OBJECTS), len(SIGNALS)), np.nan)
    answer_key = np.full((ROUNDS, len(OBJECTS), ANSWER_KEY_REPEATS, ANSWER_KEY_ROUNDS), np.nan)

    for round_number in range(ROUNDS):
        on_table = APPEARS_AT <= round_number

        # Push at random, and do every push twice.
        pushes = random_pushes(PUSHES_PER_ROUND, SHARE * on_table, rng)
        inputs = as_inputs(pushes)
        first_results = push(pushes, rng)
        second_results = push(pushes, rng)

        # Split the pushes into two random halves. A is trained on this round. B is held out until
        # the next round, so that forward progress can see what this round's training does to B.
        half_b = rng.permutation(PUSHES_PER_ROUND) < PUSHES_PER_ROUND // 2
        half_a = ~half_b

        # Read the signals on B, before any training on it.
        predictions_before = network.predict(inputs[half_b])
        per_push = {
            "raw error": raw_error(predictions_before, first_results[half_b]),
            "slope agreement": slope_agreement(network, inputs[half_b], first_results[half_b]),
            "repeat agreement": repeat_agreement(predictions_before, first_results[half_b], second_results[half_b]),
        }
        if len(past_selves) == PAST_SELF_AGE:
            per_push["progress over time"] = progress_over_time(
                past_selves[0], network, inputs[half_b], first_results[half_b]
            )

        # Keep a snapshot of the network. The untrained network of round 0 is left out:
        # compared with that, everything looks like progress.
        if round_number > 0:
            past_selves = (past_selves + [network.copy()])[-PAST_SELF_AGE:]

        # A goes into memory now.
        memory_inputs = np.concatenate([memory_inputs, inputs[half_a], inputs[half_a]])
        memory_results = np.concatenate([memory_results, first_results[half_a], second_results[half_a]])

        # Find out by brute force where extra pushes would really help. This is the slow part.
        # It starts once all the signals can be read, so that they are judged on the same rounds.
        if "progress over time" in per_push:
            test_pushes = random_pushes(TEST_PUSHES, on_table, rng)
            for obj in np.flatnonzero(on_table):
                for repeat in range(ANSWER_KEY_REPEATS):
                    answer_key[round_number, obj, repeat] = value_of_extra_pushes(
                        network, memory_inputs, memory_results, obj, test_pushes, on_table,
                        shares=SHARE * on_table, seed=rng.integers(2**31), rounds=ANSWER_KEY_ROUNDS,
                        extra_pushes=PUSHES_PER_ROUND, pushes_per_round=PUSHES_PER_ROUND,
                        train_steps=TRAIN_STEPS, batch_size=BATCH_SIZE,
                    )

        # Train on memory, which holds A but not B.
        for _ in range(TRAIN_STEPS):
            replayed = rng.integers(0, len(memory_inputs), BATCH_SIZE)
            network.train_step(memory_inputs[replayed], memory_results[replayed])

        # Read B again. The network never trained on B, so any gain came from learning on other pushes.
        predictions_after = network.predict(inputs[half_b])
        per_push["forward progress"] = forward_progress(predictions_before, predictions_after, first_results[half_b])

        # B goes into memory now, so that it is trained on from the next round on.
        memory_inputs = np.concatenate([memory_inputs, inputs[half_b], inputs[half_b]])
        memory_results = np.concatenate([memory_results, first_results[half_b], second_results[half_b]])

        # One number per object and signal: the average over that object's pushes in B.
        for obj in np.flatnonzero(on_table):
            this_object = pushes.obj[half_b] == obj
            if this_object.sum() < 2:
                continue  # too few pushes of this object in B to say anything this round
            for name, values in per_push.items():
                signals[round_number, obj, SIGNALS.index(name)] = values[this_object].mean()

    return signals, answer_key


if __name__ == "__main__":
    with Pool() as pool:  # the runs do not depend on each other, so do them side by side
        runs = pool.map(run, SEEDS)

    Path("results").mkdir(exist_ok=True)
    np.savez(
        report_step1.RESULTS_FILE,
        signals=np.stack([signals for signals, _ in runs]),          # [run, round, object, signal]
        answer_key=np.stack([answer_key for _, answer_key in runs]),  # [run, round, object, repeat, rounds ahead]
        objects=OBJECTS, signal_names=SIGNALS, appears_at=APPEARS_AT,
    )
    report_step1.main()
