"""Step 2: let the learner choose its own pushes. Does it learn the world faster?

Each learner is the step-1 network plus a critic (critic.py): a plain head that reads
the network's last hidden layer and forecasts how good a push would be. Every round:

    1. draw 8 random candidate pushes for every push it may make, and let the critic
       score them
    2. make the best three quarters of its pushes, plus a quarter at random, so that every
       object keeps being pushed
    3. split the pushes into halves A and B, as in step 1
    4. train on memory, which now holds A but not B
    5. grade B with the learner's label, and let the critic learn from the grades
    6. put B into memory
    7. measure the true error of each object, against the world's rules

The budget. Every learner gets the same number of actions per round (256), the way a
robot pays for every action. Budget "fair": every learner makes 256 different pushes, each
done once, split into halves A and B of 128. Repeat agreement is the exception: its label
is read on half B and needs every push there done twice. So it makes 64 pushes in half B,
each done twice, and 128 pushes in half A, each done once: 192 different pushes in all.
Budget "doubled" is the older way, in which every learner did 128 pushes twice. It
favours repeat agreement, because only repeat agreement uses the copies. It is kept so
that the committed results of step 2 and of the first map can be reproduced, and so that
step A (run_step_a.py) can compare the two budgets.

The learners differ only in the label their critic learns:

    random pushing       no critic: all pushes are random, as in step 1 (wood gets 40%)
    even pushing         no critic: all pushes are random, spread evenly over the objects.
                         A fairer baseline: part of any gain may come from just not
                         pushing wood so often.
    raw error            the floor
    repeat agreement     error that comes back when a push is repeated
    forward progress     how much easier the held-out pushes got during the round
    random grades        a control, added in step A: every push gets a random grade, so
                         the critic learns nothing real. Whatever this learner gains over
                         even pushing comes from how pushes are chosen, not from a label.

Every learner runs with two training settings: 100 steps per round, where the network
memorizes the toy's random results, and 20 steps per round, where it memorizes much less.

The map (run_map.py) uses the same run function in bigger and narrow worlds, adds the two
other labels, progress over time and slope agreement, and can switch on a guard against
memorizing noise: "early stopping" (see train below) or "predicts its noise" (a network
that also guesses how scattered each result is, see network.py). Step B adds a third guard,
"noise from repeats": a second network learns how scattered each result is from pushes
done twice only (see repeat_noise), and the learner's misses cost less where it expects
scatter.

Run it with:   python3 run_step2.py
It takes about twenty seconds, writes results/step2.npz, and prints the report. It uses
the "doubled" budget, as step 2 did.
"""

from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import report_step2
from answer_key import true_error_per_object
from critic import Critic, forecast_agreement, object_average
from network import LOG_VARIANCE_RANGE, Network
from run_step1 import BATCH_SIZE, PAST_SELF_AGE, PUSHES_PER_ROUND, ROUNDS, SEEDS, SHARE, TEST_PUSHES
from signals import forward_progress, progress_over_time, raw_error, repeat_agreement, slope_agreement
from world import RESULT_SIZE, TABLE, joined

BASELINES = ("random pushing", "even pushing")  # learners without a critic
LEARNERS = (*BASELINES, "raw error", "repeat agreement", "forward progress")
TRAINING = (100, 20)     # training steps per round
ACTIONS_PER_ROUND = 2 * PUSHES_PER_ROUND  # 256: what every learner may spend each round
CANDIDATES_PER_PUSH = 8  # random candidates the critic scores, per push it may make
LEFT_TO_CHANCE = 0.25    # share of the pushes that stays random, so that every object keeps being pushed

# The critic check (see check_critic): two forecasters, each scored on all pushes and without the toys.
CRITIC_CHECKS = ("critic", "object_average", "critic_no_toys", "object_average_no_toys")

# Guards against memorizing noise, used by the map.
GUARDS = ("none", "early stopping", "predicts its noise")

# Early stopping: each round, a few pushes of half A go into a validation set that is never
# trained on. Training checks the error on that set every few steps, and stops the round
# once the error stops falling.
VALIDATION_PER_ROUND = 16  # records per round kept for validation: 8 pushes done twice, or 16 done once

# How a learner spends its actions each round: (pushes in half A, done twice?, pushes in half B, done twice?).
EVERY_PUSH_TWICE = (64, True, 64, True)    # the "doubled" budget
EVERY_PUSH_ONCE = (128, False, 128, False)  # the "fair" budget
B_TWICE = (128, False, 64, True)            # the "fair" budget for repeat agreement
CHECK_EVERY = 10          # training steps between two checks
PATIENCE = 2              # checks in a row without improvement before the round's training stops


def run(learner, train_steps, seed, world=TABLE, share=SHARE, guard="none", budget="fair"):
    """One learner, one training setting, one seed, in one world.

    share says how often each object is picked when pushes are left to chance. guard is
    one of GUARDS, or "noise from repeats" (added in step B, and kept out of GUARDS so that
    the map stays as it was). budget is "fair" or "doubled" (see the top of this file).
    Returns, for every round: the true error of every object, how many pushes went to every
    object, the number of training steps actually taken, and the critic check (a line over
    the rounds for each name in CRITIC_CHECKS, see check_critic).
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

    # How this learner spends its actions (see the top of this file). The guard "noise from
    # repeats" needs every push twice: its noise model learns from every pair.
    if budget == "doubled" or guard == "noise from repeats":
        a_pushes, a_twice, b_pushes, b_twice = EVERY_PUSH_TWICE
    elif learner == "repeat agreement":
        a_pushes, a_twice, b_pushes, b_twice = B_TWICE
    else:
        a_pushes, a_twice, b_pushes, b_twice = EVERY_PUSH_ONCE
    distinct_pushes = a_pushes + b_pushes

    true_error = np.full((ROUNDS, len(world.names)), np.nan)
    pushes_made = np.zeros((ROUNDS, len(world.names)))
    steps_taken = np.zeros(ROUNDS)
    critic_check = {name: np.full(ROUNDS, np.nan) for name in CRITIC_CHECKS}
    past_grades = []  # (objects, grades) of the rounds the critic remembers, for the critic check

    # The guard "noise from repeats" trains a noise model on pairs: a push, and its scatter judged from its two results.
    noise_model = None
    if guard == "noise from repeats":
        noise_model = Network(rng, sizes=(world.input_size, 128, 128, 128, RESULT_SIZE))
    pair_inputs = np.zeros((0, world.input_size))
    pair_noise = np.zeros((0, RESULT_SIZE))

    for round_number in range(ROUNDS):
        on_table = world.appears_at <= round_number

        # Choose the pushes and make them.
        by_chance = np.ones(distinct_pushes, dtype=bool)  # which pushes were left to chance
        if learner == "random pushing":
            pushes = world.random_pushes(distinct_pushes, share * on_table, rng)
        elif learner == "even pushing":
            pushes = world.random_pushes(distinct_pushes, on_table, rng)
        else:
            pushes, by_chance = choose_pushes(critic, network, world, on_table, share, rng, distinct_pushes)
        pushes_made[round_number] = np.bincount(pushes.obj, minlength=len(world.names))
        inputs = world.as_inputs(pushes)
        first_results = world.push(pushes, rng)
        # The second result of every push. Only the halves done twice use it: for a half done
        # once it is thrown away unseen, as if that push had been made only once.
        second_results = world.push(pushes, rng) if a_twice or b_twice else None

        # Half A is trained on this round. Half B is held out until the next round, as in step 1.
        half_b = rng.permutation(distinct_pushes) < b_pushes
        half_a = ~half_b
        second_a = second_results if a_twice else None
        second_all_b = second_results if b_twice else None
        inputs_b, first_b, objects_b = inputs[half_b], first_results[half_b], pushes.obj[half_b]
        second_b = second_results[half_b] if b_twice else None
        predictions_before = network.predict(inputs_b)
        forecasts = critic.score(network, inputs_b)  # what the critic expects of B, before any grade is known

        # Every label but forward progress is read now, before any training.
        grades = grade_before_training(learner, network, past_selves, predictions_before, inputs_b, first_b, second_b)
        if learner == "random grades":
            grades = rng.normal(size=len(inputs_b))

        # Keep a snapshot for "progress over time", as in step 1.
        if round_number > 0:
            past_selves = (past_selves + [network.copy()])[-PAST_SELF_AGE:]

        # With early stopping, a few random pushes of A go into the validation set instead of memory.
        early_stopping = guard == "early stopping"
        if early_stopping:
            kept = rng.choice(np.flatnonzero(half_a), VALIDATION_PER_ROUND // (2 if a_twice else 1), replace=False)
            half_a[kept] = False
            kept_inputs, kept_results = records(inputs, first_results, second_a, kept)
            validation_inputs = np.concatenate([validation_inputs, kept_inputs])
            validation_results = np.concatenate([validation_results, kept_results])

        a_inputs, a_results = records(inputs, first_results, second_a, half_a)
        memory_inputs = np.concatenate([memory_inputs, a_inputs])
        memory_results = np.concatenate([memory_results, a_results])
        validation = (validation_inputs, validation_results) if early_stopping else None
        if noise_model is not None:
            pair_inputs = np.concatenate([pair_inputs, inputs[half_a]])
            pair_noise = np.concatenate([pair_noise, repeat_noise(first_results[half_a], second_results[half_a])])
            train(noise_model, pair_inputs, pair_noise, train_steps, rng)
        steps_taken[round_number] = train(
            network, memory_inputs, memory_results, train_steps, rng, validation, noise_model,
        )

        # Forward progress needs the network after training.
        if learner == "forward progress":
            grades = forward_progress(predictions_before, network.predict(inputs_b), first_b)

        # The critic check, once there are grades and the critic has learned from earlier ones.
        if grades is not None and past_grades:
            chance_b = by_chance[half_b]
            check = check_critic(forecasts[chance_b], grades[chance_b], objects_b[chance_b], world, past_grades)
            for name in CRITIC_CHECKS:
                critic_check[name][round_number] = check[name]

        # Then the critic learns from the grades.
        if grades is not None:
            critic.learn(network, inputs_b, grades)
            past_grades = (past_grades + [(objects_b, grades)])[-critic.rounds_remembered:]

        # B goes into memory now, so that it is trained on from the next round on.
        b_inputs, b_results = records(inputs, first_results, second_all_b, half_b)
        memory_inputs = np.concatenate([memory_inputs, b_inputs])
        memory_results = np.concatenate([memory_results, b_results])
        if noise_model is not None:
            pair_inputs = np.concatenate([pair_inputs, inputs_b])
            pair_noise = np.concatenate([pair_noise, repeat_noise(first_b, second_b)])

        true_error[round_number] = true_error_per_object(network, test_pushes, on_table, world)

    return true_error, pushes_made, steps_taken, critic_check


def records(inputs, first_results, second_results, which):
    """What goes into memory for the pushes picked out by which: one record per result.

    second_results is None for pushes done once."""
    if second_results is None:
        return inputs[which], first_results[which]
    return (np.concatenate([inputs[which], inputs[which]]),
            np.concatenate([first_results[which], second_results[which]]))


def check_critic(forecasts, grades, objects, world, past_grades):
    """How well the critic forecast the grades of B's pushes that were left to chance.

    Only those pushes count: the critic's job is to rank random candidates, and its own
    picks would only show how it ranks pushes it already likes.

    Next to the critic is the object average (critic.py): a forecaster that knows only which
    object a push is for. If it does as well as the critic, the critic has learned no more
    than which objects are good. Both are scored on all these pushes, and again without the
    toys, because the toys' random results can decide the order on their own.
    Returns a forecast_agreement for each name in CRITIC_CHECKS.
    """
    past_objects = np.concatenate([round_objects for round_objects, _ in past_grades])
    past = np.concatenate([round_grades for _, round_grades in past_grades])
    average = object_average(objects, past_objects, past)
    no_toys = np.asarray(world.kinds)[objects] != "toy"
    return {
        "critic": forecast_agreement(forecasts, grades),
        "object_average": forecast_agreement(average, grades),
        "critic_no_toys": forecast_agreement(forecasts[no_toys], grades[no_toys]),
        "object_average_no_toys": forecast_agreement(average[no_toys], grades[no_toys]),
    }


def train(network, memory_inputs, memory_results, train_steps, rng, validation=None, noise_model=None):
    """Train for up to train_steps steps on batches drawn from memory. Returns the steps taken.

    validation, if given, is (inputs, results) of pushes that are never trained on. Their
    error is checked every CHECK_EVERY steps, and training stops once it has failed to
    improve PATIENCE checks in a row. This is the standard guard against memorizing,
    called early stopping.

    noise_model, if given, guesses how scattered each result is (see expected_noise). A
    miss then costs less where more scatter is expected (see Network.backward).
    """
    if validation is not None:
        best, misses = validation_error(network, *validation), 0
    for step in range(1, train_steps + 1):
        replayed = rng.integers(0, len(memory_inputs), BATCH_SIZE)
        noise = None if noise_model is None else expected_noise(noise_model, memory_inputs[replayed])
        network.train_step(memory_inputs[replayed], memory_results[replayed], noise)
        if validation is not None and step % CHECK_EVERY == 0:
            error = validation_error(network, *validation)
            if error < best:
                best, misses = error, 0
            else:
                misses += 1
                if misses == PATIENCE:
                    return step
    return train_steps


def repeat_noise(first_results, second_results):
    """How scattered each push's result is, judged from the push done twice: half the squared difference.

    For a push whose result never changes this is exactly 0, however rare or sharp its
    behaviour. Averaged over many pushes alike, it is the variance of their results.
    """
    return (first_results - second_results) ** 2 / 2


def expected_noise(noise_model, inputs):
    """The noise model's guess of each result's variance, never below the smallest variance
    the noise-predicting network can guess, so that no miss can cost without limit."""
    return np.maximum(noise_model.predict(inputs), 0) + np.exp(LOG_VARIANCE_RANGE[0])


def validation_error(network, inputs, results):
    """The average squared miss on pushes the network has never trained on."""
    return raw_error(network.predict(inputs), results).mean()


def choose_pushes(critic, network, world, on_table, share, rng, count):
    """The critic's favourite candidates, plus a few pushes left to chance.

    Returns the count pushes, and a yes/no per push: was it left to chance?
    """
    # Candidates come from every object on the table equally often, so the critic alone decides the mix.
    candidates = world.random_pushes(CANDIDATES_PER_PUSH * count, on_table, rng)
    best_first = np.argsort(-critic.score(network, world.as_inputs(candidates)))
    chance_count = round(LEFT_TO_CHANCE * count)
    chosen = candidates.take(best_first[:count - chance_count])
    left_to_chance = world.random_pushes(chance_count, share * on_table, rng)
    return joined(chosen, left_to_chance), np.arange(count) >= count - chance_count


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


def run_doubled(learner, train_steps, seed):
    """A run with the "doubled" budget, as step 2 used."""
    return run(learner, train_steps, seed, budget="doubled")


if __name__ == "__main__":
    jobs = list(product(LEARNERS, TRAINING, SEEDS))
    with Pool() as pool:  # the runs do not depend on each other, so do them side by side
        results = pool.starmap(run_doubled, jobs)

    # Arrays are [learner, training setting, run, round, object].
    shape = (len(LEARNERS), len(TRAINING), len(SEEDS), ROUNDS, len(TABLE.names))
    Path("results").mkdir(exist_ok=True)
    np.savez(
        report_step2.RESULTS_FILE,
        true_error=np.array([result[0] for result in results]).reshape(shape),
        pushes_made=np.array([result[1] for result in results]).reshape(shape),
        learners=LEARNERS, training=TRAINING, objects=TABLE.names, appears_at=TABLE.appears_at, share=SHARE,
    )
    report_step2.main()
