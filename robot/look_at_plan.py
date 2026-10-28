"""What the planner in test 2 sees: why it barely tells a good world model from an untrained one.

Planning with the demo-trained model succeeds about as often as planning with the runs' untrained
round-0 learner (plan_demo_trained.txt, plan_untrained.txt). Either the model rarely imagines any
plan reaching success, and then the planner does a random plan; or it imagines success, but the plan
it chose fails in the true world. For each planning call, on the same starts and seeds as plan.py,
this records both: did the model imagine the chosen plan reaching success, and does the chosen plan,
done in full in the true world from the same state, reach success?

Run it with:   ../.venv/bin/python look_at_plan.py      (a few minutes, on the CPU)
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import plan
from kinematics import HandFinder
from sim import Sim
from world_model import WorldModel

RESULTS = Path(__file__).resolve().parent / "results"
MODELS = ("demo_trained.pt", "untrained.pt")
PER_K = 20


def one_try(task):
    """plan.one_try with a world model, also giving, for each planning call, (imagined success, real success)."""
    step, k, seed, checkpoint = task
    demos, carried = np.load(plan.DEMOS / "demo_states.npz"), np.load(plan.DEMOS / "test_steps.npz")
    real, check = Sim(), Sim()
    torch.set_num_threads(1)
    learned = WorldModel(dict(np.load(RESULTS / "scales.npz")))
    learned.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    finder, peg = HandFinder(real.model), torch.tensor(real.square_peg)
    plan.put(real, (demos["states"][step - k], carried["seen"][step - k][16:18], np.zeros(real.model.nv)))
    rng = np.random.default_rng(seed)
    horizon = min(k + 5, 35)
    chosen, calls = None, []
    for _ in range(0, k + plan.EXTRA_STEPS, plan.DO_BEFORE_REPLANNING):
        taken, state = plan.take(real), real.state()
        mean = None if chosen is None else np.concatenate([chosen[plan.DO_BEFORE_REPLANNING:],
                                                           np.zeros((plan.DO_BEFORE_REPLANNING, 7))])
        score = lambda plans: plan.score_in_world_model(learned, finder, peg, state, plans)
        chosen = plan.cross_entropy(score, horizon, rng, mean)
        calls.append((score(chosen[None])[0] > 0, plan.score_in_true_world(check, taken, chosen[None])[0] > 0))
        for action in chosen[:plan.DO_BEFORE_REPLANNING]:
            real.step(action)
            if plan.succeeded(real):
                return k, True, calls
    return k, False, calls


def main():
    found = plan.success_steps()
    steps = [found[i] for i in plan.HELD_OUT if found[i] is not None][:PER_K]
    for name in MODELS:
        tasks = [(step, k, 1000 * k + i, str(RESULTS / name)) for k in plan.KS for i, step in enumerate(steps)]
        with Pool(plan.WORKERS) as pool:
            results = pool.map(one_try, tasks)
        print(f"Planning with {name} ({PER_K} demo starts per k):")
        for k in plan.KS:
            tries = [(won, calls) for kk, won, calls in results if kk == k]
            calls = [call for _, these in tries for call in these]
            imagined = [real for dreamed, real in calls if dreamed]
            not_imagined = [real for dreamed, real in calls if not dreamed]
            print(f"   k = {k:2d}: {sum(won for won, _ in tries):2d} of {len(tries)} tries succeeded."
                  f" Planning calls: {len(calls)}; the chosen plan imagined to succeed in {len(imagined)},"
                  f" and of those it succeeded in the true world in {sum(imagined)};"
                  f" of the other {len(not_imagined)}, it succeeded anyway in {sum(not_imagined)}")


if __name__ == "__main__":
    main()
