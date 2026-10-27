"""The toy comparison: every candidate way of choosing pushes, run end to end.

Every learner gets the same 256 actions per round, the same network trained the same way
(50 steps per round), and the same way of picking: 8 random candidate pushes for every push
it may make, its best-scored three quarters, and a quarter left to chance so that every
object keeps being pushed. Learners differ only in how they score a candidate:

    with a critic   a neural critic (critic.NeuralCritic) forecasts a label that is read
                    after acting, such as raw error or repeat agreement
    directly        a measure read before acting scores the candidate itself, with no
                    critic, as the established methods do: steepness, ensemble disagreement,
                    random-network novelty, feature novelty, pinned weights, prediction drift
    combined        the candidate's place among the round's candidates on that score (from
                    0 to 1), times its place on steepness: it scores high only if it is high
                    on both (see combined)

Labels that need a second look at a push pay for it out of the same 256 actions. Repeat-
based labels make their held-out pushes twice: 128 pushes in half A, and 64 in half B done
twice. Near copies make 4 close copies of every held-out push: 96 pushes in half A, 32 in
half B, and 128 copies.

LEARNERS lists the learners in the four groups of the design agreed on 2026-10-20: our
bets, the bar to beat, baselines, and bookkeeping.

Worlds: random worlds of 6 and 24 objects, plain and narrow (run_map.build_world), 10 runs
each. Every learner of a run shares that run's world.

Run it with:   python3 run_comparison.py      (about 15 minutes on the M4 Pro)
It writes results/comparison.npz and prints the report (report_comparison.py).
"""

from dataclasses import dataclass, field
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import report_comparison
from answer_key import true_error_per_object
from critic import Critic, NeuralCritic
from measures import (
    ensemble_disagreement, feature_novelty, pinned_weight_uncertainty, prediction_drift, random_network_novelty,
    steepness,
)
from network import Network
from run_map import build_world
from run_step1 import PAST_SELF_AGE, ROUNDS, TEST_PUSHES
from run_step2 import CANDIDATES_PER_PUSH, LEFT_TO_CHANCE, train
from signals import (
    forward_progress, last_layer_repeated, progress_over_time, raw_error, region_average, repeat_agreement,
    similar_pushes, slope_agreement, slope_agreement_repeated, slope_agreement_with_copies,
    slope_agreement_with_similar, slope_agreements, slope_consistency, teaching_value,
)
from world import RESULT_SIZE, joined, near_copies

WORLDS = ("6", "24", "narrow 6", "narrow 24")
SEEDS = tuple(range(10))
TRAIN_STEPS = 50
NUDGE = 0.02          # how far a near copy is from its push: 2% of each setting's range
COPIES = 4            # near copies made of every held-out push
SNAPSHOTS = 10        # copies of the network kept from earlier rounds (progress over 10 rounds needs 10)


@dataclass(frozen=True)
class Plan:
    """How one learner scores candidate pushes, and what it pays for that."""

    group: str                     # "our bets", "bar to beat", "baselines" or "bookkeeping"
    label: str = None              # what its critic learns, read after acting (None: no critic)
    measure: str = None            # what scores candidates directly, read before acting
    times_steepness: bool = False  # combined with steepness (see combined)
    critic: str = "neural"         # "neural", or "line" for the straight-line check
    repeats: bool = False          # its held-out pushes are done twice
    copies: int = 0                # near copies made of every held-out push
    new_first: bool = False        # pushes on objects first pushed fewer than 3 rounds ago come first


LEARNERS = {
    # Our bets: steepness, alone and combined with a partner that fades once a place is learned.
    "steepness": Plan("our bets", measure="steepness"),
    "raw error x steepness": Plan("our bets", label="raw error", times_steepness=True),
    "repeat agreement x steepness": Plan("our bets", label="repeat agreement", times_steepness=True, repeats=True),
    "disagreement x steepness": Plan("our bets", measure="disagreement", times_steepness=True),
    # The bar to beat: the bets' other parents, and the established methods.
    "raw error": Plan("bar to beat", label="raw error"),
    "repeat agreement": Plan("bar to beat", label="repeat agreement", repeats=True),
    "ensemble disagreement": Plan("bar to beat", measure="disagreement"),
    "random-network novelty": Plan("bar to beat", measure="novelty"),
    "feature novelty": Plan("bar to beat", measure="feature novelty"),
    "pinned weights": Plan("bar to beat", measure="pinned weights"),
    # Baselines: the lines everything is read against.
    "not choosing": Plan("baselines"),
    "do-nothing critic": Plan("baselines", label="random"),
    # Bookkeeping: expected to lose, run to close the loop.
    "slope, same push twice": Plan("bookkeeping", label="slope twice", repeats=True),
    "slope, same push twice, last layer": Plan("bookkeeping", label="slope twice, last layer", repeats=True),
    "near copies 2%": Plan("bookkeeping", label="near copies", copies=COPIES),
    "forward progress": Plan("bookkeeping", label="forward progress"),
    "progress over 3 rounds": Plan("bookkeeping", label="progress 3"),
    "progress over 10 rounds": Plan("bookkeeping", label="progress 10"),
    "progress, new first": Plan("bookkeeping", label="progress 3", new_first=True),
    "learning progress by region": Plan("bookkeeping", label="region progress"),
    "prediction drift": Plan("bookkeeping", measure="drift"),
    "slope vs batch": Plan("bookkeeping", label="slope vs batch"),
    "slope vs neighbours": Plan("bookkeeping", label="slope vs neighbours"),
    "slope consistency": Plan("bookkeeping", label="slope consistency"),
    "teaching value, one step": Plan("bookkeeping", label="teaching value"),
    "novelty x steepness": Plan("bookkeeping", measure="novelty", times_steepness=True),
    "raw error, line critic": Plan("bookkeeping", label="raw error", critic="line"),
    "repeat agreement, line critic": Plan("bookkeeping", label="repeat agreement", critic="line", repeats=True),
}


def halves(plan):
    """(pushes in half A, pushes in half B) for this learner's 256 actions."""
    if plan.repeats:
        return 128, 64   # 128 + 64 x 2
    if plan.copies:
        return 96, 32    # 96 + 32 + 32 x 4 copies
    return 128, 128


@dataclass
class Learner:
    """Everything one learner has: its networks, its critic, and what it has seen."""

    network: Network
    settings_from: int                               # where the push's settings start in the inputs
    rng: np.random.Generator
    critic: object = None
    others: list = field(default_factory=list)       # the rest of the ensemble (disagreement)
    target: Network = None                           # random-network novelty: the fixed random network,
    imitator: Network = None                         # and the network that learns to imitate it
    memory_inputs: np.ndarray = None
    memory_results: np.ndarray = None
    seen: np.ndarray = None                          # every distinct push made so far
    past: list = field(default_factory=list)         # copies of the network from earlier rounds, newest last
    first_pushed: np.ndarray = None                  # the round each object was first pushed in (-1: not yet)


def run(learner_name, world_name, seed):
    """One learner, one world, one run. Returns lines over the rounds (see the end of this function)."""
    plan = LEARNERS[learner_name]
    rng = np.random.default_rng(seed)
    world = build_world(world_name, seed)
    sizes = (world.input_size, 128, 128, 128, RESULT_SIZE)
    me = Learner(Network(rng, sizes), settings_from=len(world.names), rng=rng)
    test_pushes = world.test_pushes(TEST_PUSHES, rng)  # drawn here, so that every learner of a run shares them
    if plan.measure == "disagreement":
        me.others = [Network(rng, sizes) for _ in range(4)]
    if plan.measure == "novelty":
        small = (world.input_size, 64, 64, 16)
        me.target, me.imitator = Network(rng, small), Network(rng, small)
    if plan.label:
        critic_rng = np.random.default_rng([seed, 1])  # its own random numbers, unlike those of any run
        me.critic = NeuralCritic(critic_rng) if plan.critic == "neural" else Critic(10, 100.0)
    me.memory_inputs, me.memory_results = np.zeros((0, world.input_size)), np.zeros((0, RESULT_SIZE))
    me.seen = np.zeros((0, world.input_size))
    me.first_pushed = np.full(len(world.names), -1)
    a_count, b_count = halves(plan)
    toy = np.array([kind == "toy" for kind in world.kinds])
    from_start = ~toy & (world.appears_at == 0)
    learnable, from_start_error, toy_error, toy_share = (np.full(ROUNDS, np.nan) for _ in range(4))

    for round_number in range(ROUNDS):
        on_table = world.appears_at <= round_number

        # Choose this round's pushes and make them.
        count = a_count + b_count
        if learner_name == "not choosing":
            pushes, by_chance = world.random_pushes(count, on_table, rng), np.ones(count, dtype=bool)
        else:
            pushes, by_chance = choose(plan, me, world, on_table, count, round_number)
        never_pushed = me.first_pushed < 0
        me.first_pushed[never_pushed & np.isin(np.arange(len(world.names)), pushes.obj)] = round_number
        inputs, first = world.as_inputs(pushes), world.push(pushes, rng)

        # Half A is trained on this round; half B is held out and graded, and may be repeated or copied.
        half_b = rng.permutation(count) < b_count
        pushes_b, inputs_b, first_b = pushes.take(half_b), inputs[half_b], first[half_b]
        second_b = world.push(pushes_b, rng) if plan.repeats else None
        copies = near_copies(pushes_b, NUDGE, rng, plan.copies) if plan.copies else None
        copy_inputs = world.as_inputs(copies) if plan.copies else None
        copy_results = world.push(copies, rng) if plan.copies else None

        before = me.network.predict(inputs_b)
        grades = label_before_training(plan, me, before, inputs_b, first_b, second_b, copy_inputs, copy_results,
                                       by_chance[half_b], pushes_b.obj)
        if round_number > 0:
            me.past = (me.past + [me.network.copy()])[-SNAPSHOTS:]

        # Learn from half A.
        a = ~half_b
        me.memory_inputs = np.concatenate([me.memory_inputs, inputs[a]])
        me.memory_results = np.concatenate([me.memory_results, first[a]])
        me.seen = np.concatenate([me.seen, inputs[a]])
        for network in (me.network, *me.others):
            train(network, me.memory_inputs, me.memory_results, TRAIN_STEPS, rng)
        if me.imitator is not None:
            train(me.imitator, me.seen, me.target.predict(me.seen), TRAIN_STEPS, rng)

        # Forward progress needs the network after training. Then the critic learns.
        if plan.label == "forward progress":
            grades = forward_progress(before, me.network.predict(inputs_b), first_b)
        if me.critic is not None and grades is not None:
            me.critic.learn(me.network, inputs_b, standardized(grades))

        # Half B, its repeats and its copies go into memory now.
        extra_inputs = [inputs_b] + ([inputs_b] if plan.repeats else []) + ([copy_inputs] if plan.copies else [])
        extra_results = [first_b] + ([second_b] if plan.repeats else []) + ([copy_results] if plan.copies else [])
        me.memory_inputs = np.concatenate([me.memory_inputs, *extra_inputs])
        me.memory_results = np.concatenate([me.memory_results, *extra_results])
        me.seen = np.concatenate([me.seen, inputs_b] + ([copy_inputs] if plan.copies else []))

        error = true_error_per_object(me.network, test_pushes, on_table, world)
        learnable[round_number] = np.nanmean(error[~toy])
        from_start_error[round_number] = error[from_start].mean()
        toy_error[round_number] = np.nanmean(error[toy]) if (toy & on_table).any() else np.nan
        toy_share[round_number] = toy[pushes.obj].mean()

    return {
        "learnable": learnable,                # true error, all objects on the table but the toys
        "from_start": from_start_error,        # the same, for the objects on the table from the first round only
        "toy_error": toy_error,                # true error of the toys: how much noise was memorized
        "toy_share": toy_share,                # share of the pushes that went to toys
    }


def choose(plan, me, world, on_table, count, round_number):
    """The learner's best-scored candidates, plus a quarter of the pushes left to chance.

    Returns the pushes, and a yes/no per push: was it left to chance?
    """
    candidates = world.random_pushes(CANDIDATES_PER_PUSH * count, on_table, me.rng)
    inputs = world.as_inputs(candidates)
    score = me.critic.score(me.network, inputs) if plan.label else measured(plan.measure, me, inputs)
    if plan.times_steepness:
        score = combined(score, steepness(me.network, inputs, me.settings_from))
    assert np.isfinite(score).all(), "a score that is not a number would leave the choice to the candidates' order"
    if plan.new_first:  # the high start: an object first pushed fewer than 3 rounds ago comes first
        pushed_at = me.first_pushed[candidates.obj]
        score = np.where((pushed_at < 0) | (round_number - pushed_at < PAST_SELF_AGE), np.inf, score)
    chance_count = round(LEFT_TO_CHANCE * count)
    best = candidates.take(np.argsort(-score, kind="stable")[:count - chance_count])
    left_to_chance = world.random_pushes(chance_count, on_table, me.rng)
    return joined(best, left_to_chance), np.arange(count) >= count - chance_count


def measured(measure, me, inputs):
    """A measure read before acting, for each candidate. Zero while there is nothing to read it from."""
    nothing = np.zeros(len(inputs))
    if measure == "steepness":
        return steepness(me.network, inputs, me.settings_from)
    if measure == "disagreement":
        return ensemble_disagreement([me.network, *me.others], inputs)
    if measure == "novelty":
        return random_network_novelty(me.target, me.imitator, inputs)
    if measure == "feature novelty":
        if not len(me.seen):
            return nothing
        parts = np.array_split(inputs, max(1, len(inputs) // 256))  # 256 candidates at a time keeps the distances small
        return np.concatenate([feature_novelty(me.network, part, me.seen) for part in parts])
    if measure == "pinned weights":
        if not len(me.memory_inputs):
            return nothing
        return pinned_weight_uncertainty(me.network, inputs, me.memory_inputs, me.memory_results, me.rng)
    if measure == "drift":
        if len(me.past) < PAST_SELF_AGE:
            return nothing
        return prediction_drift(me.past[-PAST_SELF_AGE], me.network, inputs)
    raise ValueError(measure)


def combined(score, other):
    """Each candidate's place among the candidates on score (from 0 to 1), times its place on other."""
    return rankdata(score) / len(score) * rankdata(other) / len(other)


def label_before_training(plan, me, before, inputs, first, second, copy_inputs, copy_results, yardstick, objects):
    """The grades of half B for this learner's critic, if its label can be read before training.

    Returns nothing (None) for learners without a critic, for forward progress (read after
    training), and for progress labels while the network has no copy that old yet.
    """
    label, network = plan.label, me.network
    if label == "random":
        return me.rng.normal(size=len(inputs))
    if label == "raw error":
        return raw_error(before, first)
    if label == "repeat agreement":
        return repeat_agreement(before, first, second)
    if label == "slope twice":
        return slope_agreement_repeated(network, inputs, first, second)
    if label == "slope twice, last layer":
        return last_layer_repeated(network, inputs, first, second)
    if label == "near copies":
        return slope_agreement_with_copies(network, inputs, first, copy_inputs, copy_results)
    if label in ("progress 3", "region progress") and len(me.past) >= PAST_SELF_AGE:
        progress = progress_over_time(me.past[-PAST_SELF_AGE], network, inputs, first)
        return region_average(progress, objects) if label == "region progress" else progress
    if label == "progress 10" and len(me.past) == SNAPSHOTS:
        return progress_over_time(me.past[0], network, inputs, first)
    if label == "slope vs batch":
        return slope_agreement(network, inputs, first)
    if label in ("slope vs neighbours", "slope consistency"):
        agreement, similar = slope_agreements(network, inputs, first), similar_pushes(network, inputs)
        if label == "slope vs neighbours":
            return slope_agreement_with_similar(agreement, similar)
        return slope_consistency(agreement, similar)
    if label == "teaching value":
        return teaching_value(slope_agreements(network, inputs, first), yardstick)
    return None


def standardized(grades):
    """Grades with average 0 and spread 1 within the round, so that rounds with big or small grades count alike."""
    spread = grades.std()
    return (grades - grades.mean()) / spread if spread > 0 else grades * 0


SLOW_FIRST = ("ensemble disagreement", "disagreement x steepness", "feature novelty", "pinned weights")


if __name__ == "__main__":
    # The slow learners go first, so that the cores stay busy to the end.
    cells = sorted(product(LEARNERS, WORLDS, SEEDS), key=lambda cell: cell[0] not in SLOW_FIRST)
    with Pool() as pool:
        finished = dict(zip(cells, pool.starmap(run, cells, chunksize=1)))

    # Every line is saved as [learner, world, run, round].
    def saved(line):
        return np.array([[[finished[(learner, world, seed)][line] for seed in SEEDS] for world in WORLDS]
                         for learner in LEARNERS])

    Path("results").mkdir(exist_ok=True)
    np.savez(
        report_comparison.RESULTS_FILE,
        **{line: saved(line) for line in finished[cells[0]]},
        learners=list(LEARNERS), groups=[plan.group for plan in LEARNERS.values()], worlds=WORLDS,
    )
    report_comparison.main()
