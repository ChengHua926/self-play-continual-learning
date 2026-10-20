"""The map: when does choosing your own pushes pay?

In step 2, choosing beat an even spread of pushes when the learner trained 20 steps per
round, but not at 100. The map runs the learners of step 2, with all four labels, over a
grid of:

    training steps per round   10, 20, 50, 100, 200
    worlds                     the table of steps 1 and 2, and random worlds of 6, 24
                               and 96 objects
    guard against memorizing   none, early stopping, or a network that predicts its noise

There are two sets of worlds. "plain" is as above. "narrow" uses the same worlds made
narrow (see world.py): each object shows its behaviour only in a narrow part of the
pushes, so most random pushes teach little. That is where choosing should matter most.

The random worlds come from world.make_world: per six objects, three boxes, a roller,
a spinner and a toy, with a third of the objects arriving late. Each seed gets its own
random world, and all learners of that seed share it, so they are compared on the same
worlds. Pushes left to chance are spread evenly over the objects.

The map gave every learner 128 pushes per round, each done twice (the "doubled" budget,
see run_step2.py). That favours repeat agreement. Step A (run_step_a.py) redoes part of
the map with a fair budget.

Run it with:   python3 run_map.py            (the plain worlds)
         or:   python3 run_map.py narrow     (the narrow worlds)
Each takes about ten minutes, writes its results file, and prints the report.
"""

import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import report_map
from run_step1 import ROUNDS, SEEDS
from run_step2 import GUARDS, run
from world import TABLE, make_world, narrowed

LEARNERS = ("even pushing", "raw error", "repeat agreement", "forward progress", "progress over time", "slope agreement")
TRAINING = (10, 20, 50, 100, 200)
WORLD_SETS = {
    "plain": ("table", "6", "24", "96"),
    "narrow": ("narrow table", "narrow 6", "narrow 24", "narrow 96"),
}


def build_world(name, seed):
    """The world called name: "table", or a size such as "24", optionally starting with "narrow".

    Every seed gets its own random world.
    """
    narrow = name.startswith("narrow ")
    base = name.removeprefix("narrow ")
    if base == "table":
        return narrowed(TABLE) if narrow else TABLE
    return make_world(int(base), np.random.default_rng(10_000 + seed), narrow=narrow)


def run_in_its_world(learner, train_steps, world_name, guard, seed, budget):
    """One learner in one cell of the map, for one seed. Returns lines over the rounds.

    budget is "doubled", as the map used, or "fair", as step A uses (see run_step2.py).
    """
    world = build_world(world_name, seed)
    true_error, pushes_made, steps_taken, critic_check = run(
        learner, train_steps, seed, world=world, share=np.ones(len(world.names)), guard=guard, budget=budget,
    )
    toy = np.array([kind == "toy" for kind in world.kinds])
    return {
        "learnable": np.nanmean(true_error[:, ~toy], axis=1),  # true error, all objects but the toys
        "toy_error": np.nanmean(true_error[:, toy], axis=1),   # true error of the toys: how much noise was memorized
        "toy_share": pushes_made[:, toy].sum(axis=1) / pushes_made.sum(axis=1),
        "steps_taken": steps_taken,                            # fewer than train_steps when early stopping stopped
        **critic_check,                                        # how well the critic forecast its label (run_step2.py)
    }


if __name__ == "__main__":
    world_set = sys.argv[1] if len(sys.argv) > 1 else "plain"
    worlds = WORLD_SETS[world_set]
    cells = list(product(worlds, GUARDS, TRAINING, LEARNERS, SEEDS))
    with Pool() as pool:  # the runs do not depend on each other, so do them side by side
        results = pool.starmap(
            run_in_its_world,
            [(learner, steps, world, guard, seed, "doubled") for world, guard, steps, learner, seed in cells],
        )

    # Every line is saved as [world, guard, training setting, learner, run, round].
    shape = (len(worlds), len(GUARDS), len(TRAINING), len(LEARNERS), len(SEEDS), ROUNDS)
    Path("results").mkdir(exist_ok=True)
    np.savez(
        report_map.results_file(world_set),
        **{line: np.array([result[line] for result in results]).reshape(shape) for line in results[0]},
        learners=LEARNERS, training=TRAINING, worlds=worlds, guards=GUARDS,
    )
    report_map.main(world_set)
