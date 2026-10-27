"""Test 2: do the task with a model of the world by planning, and count successes.

The task is Square's, scored success or fail as robosuite scores it: the nut around the square peg
(its center within 3 cm of the peg's, seen from above, and less than 5 cm above the table) with the
gripper away from it (at least 4.2 cm from the nut's center). Nothing else is rewarded.

The starts are the human demos' own states, k steps before they succeeded (k = 5, 10, 15, 30): the
nearer the goal, the easier. A planner, the cross-entropy method, tries 64 action sequences inside
its model, keeps the 8 best, draws the next 64 around them, three times, and does the first 5 actions
of the best sequence in the real world; then it plans again, starting from its last plan moved on 5
steps. A sequence scores 1 if it reaches success, plus a little for reaching it sooner, else 0.
Each try gets k + 20 steps.

The model is either the true world itself (the known-answer check: how far back planning with this
reward can reach at all) or a trained world model, which imagines each plan by feeding its predicted
next state back in, step by step (its mean prediction), and checks success on the imagined states.
Either way the chosen actions are done in the true world (our CPU copy, sim.Sim), and only success
there counts.

Run it with:   ../.venv/bin/python plan.py [demos per k, default 20]                 (the true world)
          or:  ../.venv/bin/python plan.py <world model .pt> [demos per k]         (a trained one)
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
import torch

from constants import TABLE_TOP
from kinematics import HandFinder
from sim import Sim
from world_model import WorldModel

DEMOS = Path(__file__).resolve().parent.parent / "data" / "robomimic" / "square_ph"
KS = (5, 10, 15, 30)
SAMPLES, ELITES, ROUNDS = 64, 8, 3
DO_BEFORE_REPLANNING = 5
EXTRA_STEPS = 20
FIRST_SPREAD = 0.5
WORKERS = 10


def succeeded(sim):
    """robosuite's success check (NutAssembly._check_success), read off our copy of the world."""
    nut = sim.data.qpos[sim.nut_q[:3]]
    hand = sim.data.site_xpos[sim.hand_site]
    on_peg = (abs(nut[:2] - sim.square_peg) < 0.03).all() and nut[2] < TABLE_TOP + 0.05
    gripper_away = 1 - np.tanh(10 * np.linalg.norm(hand - nut)) < 0.6
    return bool(on_peg and gripper_away)


def take(sim):
    return sim.recorded(), sim.finger_targets.copy(), sim.data.qacc_warmstart.copy()


def put(sim, taken):
    flat, targets, warmstart = taken
    sim.set_recorded(flat, targets)
    sim.data.qacc_warmstart[:] = warmstart
    mujoco.mj_forward(sim.model, sim.data)


def score_in_true_world(model, taken, plans):
    """Each plan's score, trying it in the true world from the state taken."""
    scores = np.zeros(len(plans))
    for i, plan in enumerate(plans):
        put(model, taken)
        for t, action in enumerate(plan):
            model.step(action)
            if succeeded(model):
                scores[i] = 1 + (len(plan) - t) / len(plan)
                break
    return scores


def imagined_success(states, finder, peg):
    """succeeded() for imagined states [n, 33]: the nut from the state, the hand from the arm's angles."""
    nut, hand = states[:, 9:12], finder.hand(states[:, :7])
    on_peg = ((nut[:, :2] - peg).abs() < 0.03).all(dim=1) & (nut[:, 2] < TABLE_TOP + 0.05)
    return on_peg & (1 - torch.tanh(10 * (hand - nut).norm(dim=1)) < 0.6)


@torch.no_grad()
def score_in_world_model(model, finder, peg, state, plans):
    """Each plan's score, imagined by the world model from this state."""
    n, horizon = plans.shape[:2]
    states = torch.tensor(state, dtype=torch.float64)[None, None].repeat(n, 1, 1)
    actions = torch.tensor(plans, dtype=torch.float64)
    scores, done = torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.bool)
    for t in range(horizon):
        mean, _ = model(states.float(), actions[:, :t + 1].float())
        following = states[:, -1] + mean[:, -1].double() * model.change_spread.double()
        states = torch.cat([states, following[:, None]], dim=1)
        newly = imagined_success(following, finder, peg) & ~done
        scores[newly] = 1 + (horizon - t) / horizon
        done |= newly
    return scores.numpy()


def cross_entropy(score, horizon, rng, mean=None):
    """The best plan [horizon, 7] that the cross-entropy method finds for this scoring."""
    mean = np.zeros((horizon, 7)) if mean is None else mean
    spread = np.full((horizon, 7), FIRST_SPREAD)
    best, best_score = mean, -1.0
    for _ in range(ROUNDS):
        plans = np.clip(mean + spread * rng.normal(size=(SAMPLES, horizon, 7)), -1, 1)
        scores = score(plans)
        if scores.max() > best_score:
            best, best_score = plans[scores.argmax()], scores.max()
        if scores.max() > 0:                         # with no success at all, keep searching where it was
            elites = plans[np.argsort(-scores)[:ELITES]][np.sort(scores)[::-1][:ELITES] > 0]
            mean, spread = elites.mean(axis=0), elites.std(axis=0) + 0.05
    return best


def success_steps():
    """For each demo, the first step at which robosuite's check says it succeeded."""
    demos = np.load(DEMOS / "demo_states.npz")
    sim = Sim()
    found, first = [], 0
    for length in demos["lengths"]:
        for t in range(length):
            sim.set_recorded(demos["states"][first + t], np.zeros(2))
            if succeeded(sim):
                found.append(first + t)
                break
        else:
            found.append(None)
        first += length
    return found


def one_try(task):
    """One start, planned with the true world (checkpoint None) or a trained world model: did it succeed?"""
    step, k, seed, checkpoint = task
    demos, carried = np.load(DEMOS / "demo_states.npz"), np.load(DEMOS / "test_steps.npz")
    real, model = Sim(), Sim()
    if checkpoint is not None:
        torch.set_num_threads(1)
        learned = WorldModel(dict(np.load(Path(__file__).resolve().parent / "results" / "scales.npz")))
        learned.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        finder, peg = HandFinder(real.model), torch.tensor(real.square_peg)
    put(real, (demos["states"][step - k], carried["seen"][step - k][16:18], np.zeros(real.model.nv)))
    rng = np.random.default_rng(seed)
    horizon = min(k + 5, 35)
    plan = None
    for _ in range(0, k + EXTRA_STEPS, DO_BEFORE_REPLANNING):
        taken = take(real)
        mean = None if plan is None else np.concatenate([plan[DO_BEFORE_REPLANNING:], np.zeros((DO_BEFORE_REPLANNING, 7))])
        if checkpoint is None:
            score = lambda plans: score_in_true_world(model, taken, plans)
        else:
            score = lambda plans: score_in_world_model(learned, finder, peg, real.state(), plans)
        plan = cross_entropy(score, horizon, rng, mean)
        for action in plan[:DO_BEFORE_REPLANNING]:
            real.step(action)
            if succeeded(real):
                return k, True
    return k, False


def main(per_k, checkpoint=None):
    steps = [s for s in success_steps() if s is not None]
    tasks = [(step, k, 1000 * k + i, checkpoint) for k in KS for i, step in enumerate(steps[:per_k])]
    with Pool(WORKERS) as pool:
        results = pool.map(one_try, tasks)
    model = "the true world" if checkpoint is None else Path(checkpoint).name
    print(f"Planning with {model} as the model, success or fail only ({per_k} demo starts per k):")
    for k in KS:
        wins = [won for kk, won in results if kk == k]
        print(f"   k = {k:2d} steps before the human's success: {np.mean(wins):.0%} succeeded ({sum(wins)} of {len(wins)})")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0].endswith(".pt"):
        main(int(arguments[1]) if len(arguments) > 1 else 20, arguments[0])
    else:
        main(int(arguments[0]) if arguments else 20)
