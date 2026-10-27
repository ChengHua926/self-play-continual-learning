"""Self-play in the robot world: the paper's loop, with the simulator as the machine and a world model as
the learner.

Several runs go in lockstep on one GPU. Every round, every run makes its experiments, the GPU world
(gpu_world.py) runs all of them at once, then each run's learner, and its generator if it has one,
takes a step. The arms (Run.arm):
    self-play            the paper's generator and reward: the world model's learning progress
    shuffle              the same, with the rewards shuffled across the pool before any use
    uniform_programs     programs drawn from the prior g0, with no generator (the paper's uniform)
    plain_random         random bytes for each experiment
    peoples_way          what people write by hand: the scene's random starts, widened, then random
                         waypoints (gpu_ways.py)
    peoples_way_at_nut   the same starts, then a scripted, randomized grasp of the nut
Every run gets 1,536 experiments of 50 steps a round, and its learner one AdamW step on them, for
2,048 rounds, as in the paper. The generator, its pool (512 fresh programs, 512 edits of programs in
the archive, 512 replays), the archive and the reward are the reproduction's, unchanged
(../selfplay); the programs' bytes become experiments by robot_machine's rules.

Every 256 rounds it scores each learner on test 1 (scores.py) and prints; every 128 it saves, and
started again it continues from its last save. Run settings marked GUESS were not tuned.

Run it with:   ../.venv/bin/python self_play.py <folder> [rounds] [experiments per run]
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import paths  # noqa: F401  (makes ../selfplay importable)
import gpu_ways
import world_model
from constants import STEPS
from gpu_world import GpuWorld
from robot_machine import NEEDED
from scores import TestOne

import archive as paper_archive
import machine as paper_machine
import reward as paper_reward
import train as paper
from generator import uniform_programs
from model import ByteModel

ARMS = ("self-play", "shuffle", "uniform_programs", "plain_random", "peoples_way", "peoples_way_at_nut")
SEEDS = (0, 1)
SCALES = Path(__file__).resolve().parent / "results" / "scales.npz"
SCORE_EVERY = 256
SAVE_EVERY = 128


@dataclass(frozen=True)
class Run:
    arm: str
    seed: int
    learning_rate: float = 3e-3             # GUESS: the world model's, after warmup
    warmup: int = 50                        # rounds, as in the reproduction
    generator_learning_rate: float = 2e-3   # the reproduction's (0.2 times its learner's 1e-2)
    beta: float = 0.01                      # the reproduction's GUESS: the pull toward g0
    fresh: int = 512                        # the reproduction's GUESS for the pool
    mutants: int = 512
    replay: int = 512


class RunState:
    """One run's learner, and generator if it has one, with everything they carry between rounds."""

    def __init__(self, run, scales, device):
        self.run, self.device = run, device
        torch.manual_seed(run.seed)
        self.rng = np.random.default_rng(run.seed)
        self.torch_rng = torch.Generator(device=device).manual_seed(run.seed)
        self.learner = world_model.WorldModel(scales).to(device)
        self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=run.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
        self.then = {}                      # the learner's weights after each recent round's step, kept in the
                                            # machine's memory: up to 1,024 of them, about 2 GB a run
        self.generator = None
        if run.arm in ("self-play", "shuffle"):
            generator_run = paper.Run(arm="self-play", seed=run.seed, beta=run.beta, fresh=run.fresh,
                                      mutants=run.mutants, replay=run.replay)
            self.generator = paper.new_model(generator_run, device)
            self.generator_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=run.generator_learning_rate,
                                                         betas=(0.9, 0.999), weight_decay=0.0)
            self.pool_run, self.archive, self.bank = generator_run, paper_archive.Archive(), paper_archive.ReplayBank()
        self.record = {"run": asdict(run), "rounds": {key: [] for key in (
            "loss", "reward_mean", "reward_spread", "fresh_kl", "bytes_written", "nut_moved", "nut_held_at_end",
            "nut_on_peg_at_end", "seconds")}, "scores": {}}

    def set_learning_rates(self, round_):
        warm = min(1.0, (round_ + 1) / self.run.warmup)
        for group in self.optimizer.param_groups:
            group["lr"] = self.run.learning_rate * warm
        if self.generator is not None:
            for group in self.generator_optimizer.param_groups:
                group["lr"] = self.run.generator_learning_rate * warm

    def programs(self, count):
        """This round's programs: the pool of fresh, edited and replayed ones, or draws from g0."""
        if self.generator is not None:
            return paper.build_pool(self.pool_run, self.generator, self.archive, self.bank, self.rng, self.device)
        return uniform_programs(count, self.rng), None, None

    def state(self):
        saved = {"learner": self.learner.state_dict(), "optimizer": self.optimizer.state_dict(), "then": self.then,
                 "rng": self.rng.bit_generator.state, "torch_rng": self.torch_rng.get_state(), "record": self.record}
        if self.generator is not None:
            saved.update(generator=self.generator.state_dict(), generator_optimizer=self.generator_optimizer.state_dict(),
                         niches=self.archive.niches, bank=self.bank.programs, bank_log_probs=self.bank.sampled_log_probs)
        return saved

    def load(self, saved):
        self.learner.load_state_dict(saved["learner"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.then, self.record = saved["then"], saved["record"]
        self.rng.bit_generator.state = saved["rng"]
        self.torch_rng.set_state(saved["torch_rng"].cpu())
        if self.generator is not None:
            self.generator.load_state_dict(saved["generator"])
            self.generator_optimizer.load_state_dict(saved["generator_optimizer"])
            self.archive.niches, self.bank.programs, self.bank.sampled_log_probs = saved["niches"], saved["bank"], saved["bank_log_probs"]


def measure_scales(world, device):
    """Each state number's mean and spread, and the spread of its change in one step, over random
    experiments: half plain random, half people's way. Made once; the same for every arm."""
    half = world.copies // 2
    generator = torch.Generator(device=device).manual_seed(12345)
    random_start, random_actions = gpu_ways.plain_random(half, generator, world.layout, device)
    people = gpu_ways.PeoplesWay(world, slice(half, world.copies), generator, aimed=False)
    people_start = people.start()
    world.set_starts(**{k: torch.cat([random_start[k], people_start[k]]) for k in random_start})
    states = [world.state().double()]
    for step in range(STEPS):
        world.step(torch.cat([random_actions[:, step], people.act(step)]))
        states.append(world.state().double())
    states = torch.stack(states, dim=1)
    changes = states[:, 1:] - states[:, :-1]
    scales = {"state_mean": states.mean(dim=(0, 1)), "state_spread": states.std(dim=(0, 1)).clamp_min(1e-3),
              "change_spread": changes.std(dim=(0, 1)).clamp_min(1e-4)}
    scales = {k: v.cpu().numpy() for k, v in scales.items()}
    SCALES.parent.mkdir(exist_ok=True)
    np.savez(SCALES, **scales)
    return scales


def main(folder, rounds=2049, per_run=1536):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    runs = [Run(arm, seed, fresh=per_run // 3, mutants=per_run // 3, replay=per_run - 2 * (per_run // 3))
            for arm in ARMS for seed in SEEDS]
    world = GpuWorld(len(runs) * per_run, device)
    scales = dict(np.load(SCALES)) if SCALES.exists() else measure_scales(world, device)
    states_of = [RunState(run, scales, device) for run in runs]
    rows = [slice(i * per_run, (i + 1) * per_run) for i in range(len(runs))]
    test_one = TestOne(device)
    peg = torch.tensor(world.layout.square_peg, dtype=torch.float32, device=device)
    limits = paper_machine.limits(64, 1)            # the 99k generator's machine limits, as in the reproduction
    first = 0
    if (folder / "state.pt").exists():
        saved = torch.load(folder / "state.pt", map_location="cpu", weights_only=False)     # load_state_dict moves weights to the GPU
        for state, one in zip(states_of, saved["runs"]):
            state.load(one)
        first = saved["round"] + 1
        print(f"continuing from round {first}", flush=True)

    for round_ in range(first, rounds):
        started = time.time()
        starts, plans, ways, pools = [], [], [], []
        for state, row in zip(states_of, rows):
            state.set_learning_rates(round_)
            arm = state.run.arm
            if arm in ("self-play", "shuffle", "uniform_programs"):
                programs, kinds, sampled = state.programs(per_run)
                outputs, written, depths = paper_machine.run(programs, state.rng.integers(0, 2 ** 63, size=len(programs)), *limits)
                start, actions = gpu_ways.decode(torch.from_numpy(outputs[:, 1:1 + NEEDED].copy()),
                                                 torch.from_numpy(np.minimum(written, NEEDED)), world.layout, device)
                pools.append((programs, kinds, sampled, depths, written))
                ways.append(None)
            elif arm == "plain_random":
                start, actions = gpu_ways.plain_random(per_run, state.torch_rng, world.layout, device)
                pools.append(None)
                ways.append(None)
            else:
                way = gpu_ways.PeoplesWay(world, row, state.torch_rng, aimed=(arm == "peoples_way_at_nut"))
                start, actions = way.start(), None
                pools.append(None)
                ways.append(way)
            starts.append(start)
            plans.append(actions)
        world.set_starts(**{k: torch.cat([s[k] for s in starts]) for k in starts[0]})
        states = torch.zeros(world.copies, STEPS + 1, 33, device=device)
        actions = torch.zeros(world.copies, STEPS, 7, device=device)
        states[:, 0] = world.state()
        for step in range(STEPS):
            actions[:, step] = torch.cat([plan[:, step] if plan is not None else way.act(step)
                                          for plan, way in zip(plans, ways)]).float()
            world.step(actions[:, step])
            states[:, step + 1] = world.state()
        simulated = time.time() - started

        for state, row, pool in zip(states_of, rows, pools):
            s, a = states[row], actions[row]
            loss = world_model.losses(state.learner, s[:, :-1], a, s[:, 1:]).mean()
            state.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(state.learner.parameters(), paper.CLIP)
            state.optimizer.step()
            notes = state.record["rounds"]
            notes["loss"].append(loss.item())
            nut_start, nut_end = s[:, 0, 9:12], s[:, -1, 9:12]
            notes["nut_moved"].append(((nut_end - nut_start).norm(dim=1) > 0.02).float().mean().item())
            notes["nut_held_at_end"].append((nut_end[:, 2] > 0.86).float().mean().item())
            notes["nut_on_peg_at_end"].append((((nut_end[:, :2] - peg).abs() < 0.03).all(dim=1) & (nut_end[:, 2] < 0.87)).float().mean().item())
            if state.generator is None:
                for key in ("reward_mean", "reward_spread", "fresh_kl"):
                    notes[key].append(None)
                notes["bytes_written"].append(float(np.mean(pool[4])) if pool is not None else None)
                continue
            programs, kinds, sampled, depths, written = pool
            state.then[round_] = {name: w.cpu() for name, w in paper_reward.snapshot(state.learner).items()}
            looking_back = {name: w.to(device) for name, w in state.then[round_ // 2].items()}
            along = paper_reward.direction(state.learner, state.optimizer, looking_back)
            rewards = paper_reward.alignment(state.learner, world_model.pack(s[:, :-1], a, s[:, 1:]), along, device,
                                             losses_of=world_model.packed_losses)
            if state.run.arm == "shuffle":
                rewards = rewards[torch.from_numpy(state.rng.permutation(len(rewards))).to(device)]
            now = paper.generator_step(state.generator, state.generator_optimizer, programs, kinds, sampled, rewards,
                                       state.run.beta, device)
            state.archive.decay()
            state.archive.admit(programs, depths, rewards.tolist())
            sampled_now = now.tolist()
            new = [i for i in range(len(programs)) if kinds[i] != paper.REPLAY]
            state.bank.add([programs[i] for i in new], [sampled_now[i] if kinds[i] == paper.FRESH else np.nan for i in new])
            fresh = torch.from_numpy(kinds == paper.FRESH).to(device)
            notes["reward_mean"].append(rewards.mean().item())
            notes["reward_spread"].append(rewards.std().item())
            notes["fresh_kl"].append((now - paper.prior_log_probs(programs).to(device))[fresh].mean().item())
            notes["bytes_written"].append(float(np.mean(written)))
            for old in [r for r in state.then if r < (round_ + 1) // 2]:
                del state.then[old]

        for state in states_of:
            state.record["rounds"]["seconds"].append(time.time() - started)
        if round_ % 16 == 0:
            losses = {arm: np.mean([st.record["rounds"]["loss"][-1] for st in states_of if st.run.arm == arm]) for arm in ARMS}
            print(f"round {round_}: {time.time() - started:.1f} s ({simulated:.1f} simulating); training loss "
                  + ", ".join(f"{arm} {value:.3f}" for arm, value in losses.items()), flush=True)
        if round_ % SCORE_EVERY == 0 or round_ == rounds - 1:
            for state in states_of:
                state.record["scores"][str(round_)] = test_one.score(state.learner)
                name = f"{state.run.arm}-{state.run.seed}"
                torch.save(state.learner.state_dict(), folder / f"learner_{name}_{round_:04d}.pt")
                if state.generator is not None:
                    torch.save(state.generator.state_dict(), folder / f"generator_{name}_{round_:04d}.pt")
            print(f"round {round_} test 1 (nats per number; all / holding / not holding):", flush=True)
            for state in states_of:
                scores = state.record["scores"][str(round_)]
                print(f"   {state.run.arm}-{state.run.seed}: {scores['all']:.3f} / {scores['holding']:.3f} / {scores['not holding']:.3f}", flush=True)
        if round_ % SAVE_EVERY == 0 or round_ == rounds - 1:
            torch.save({"runs": [state.state() for state in states_of], "round": round_}, folder / "state.pt.tmp")
            (folder / "state.pt.tmp").replace(folder / "state.pt")
            (folder / "record.json").write_text(json.dumps([state.record for state in states_of]))


if __name__ == "__main__":
    arguments = sys.argv[1:]
    main(arguments[0], *(int(a) for a in arguments[1:]))
