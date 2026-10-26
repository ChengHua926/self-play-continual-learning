"""One self-play run (Cowsik et al., "Self-Play Pretraining with Zero Data"), or one of the runs it is compared with.

Two models of the same shape train together. The generator writes programs, the machine runs
them, and the learner learns to predict their outputs byte by byte. Each round:
    1. Build the pool of programs: fresh ones from the generator, one-symbol edits of programs in
       the archive, and programs replayed from earlier rounds (archive.py). Slots the archive or
       the replay bank cannot fill yet go to fresh programs.
    2. Run them on the machine (machine.py). Their outputs are the learner's data.
    3. One learner step on all the outputs: AdamW on the mean next-byte loss (equation 1).
    4. Reward each program (reward.py), with the learner as it is after this step. The paper calls
       the weights it looks back to "the learner checkpoint", and the released checkpoints of
       round e hold the weights after round e's step; so here theta_e is the weights after it too.
    5. One generator step: policy gradient plus expert iteration (equations 4 and 5).
    6. Update the archive and the replay bank.

The arms (Run.arm):
    self-play    the authors' method, with the reward of reward.py
    uniform      no generator: every program is drawn from the prior g0, and nothing is replayed
    shuffle      the same reward, shuffled across the pool before any use, so it no longer says
                 anything about its program but keeps the same spread
    loss_delta   the reward is |loss before the learner's step - loss after|: the change over the
                 same step that the paper's last_step reward looks along

The paper does not give most training settings. These were read off the authors' released
weights for the 99k model (round 0, then every 256 rounds):
    - The first learner step moved every norm weight by exactly 2.0e-4, and the first generator
      step by exactly 4.0e-5. Adam's first step moves each weight by the learning rate, so at
      round 0 the learning rates were 2.0e-4 and 4.0e-5: the generator's is 0.2 times the learner's.
    - After warmup the generator's weights moved up to 1.2e-3 per round. Adam moves a weight by at
      most about its learning rate per step (not a strict limit), so the generator's rate is at least
      about that, and the learner's at least about 6e-3. Learning rates are usually tried as 1 or 3
      times a power of 10, so the learner's is taken as 1e-2, reached by a linear warmup of 50
      rounds (1e-2 / 50 = 2e-4). This one is an inference; fingerprint.py's movement check tests it.
    - The generator's output weights for bytes it may never write stayed exactly the same over 768
      rounds, so there is no weight decay.
    - The starting weights follow a normal distribution cut at 3 standard deviations. The standard
      deviation is 1 / sqrt(inputs) for each matrix and 1 / width for the embedding.
The other settings are guesses, marked GUESS.

Run it with:   ../.venv/bin/python train.py <folder> [setting=value ...] [--no-scores]
e.g.           ../.venv/bin/python train.py results/self-play-0 arm=self-play seed=0
It saves into the folder and continues from the last saved state if there is one.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before the GPU starts: repeatable matrix products
# Let the GPU's memory cache grow in place. Sampling makes slightly bigger tensors at each of up to
# 255 steps; without this, freed smaller blocks cannot be reused, and one run's cache swelled to
# nearly the whole 80 GB GPU while it used under 6 GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import machine
import reward
import score
from archive import Archive, ReplayBank, body
from generator import log_probs, prior_log_probs, sample, uniform_programs
from model import CONTEXT, VOCABULARY, ByteModel

# On a GPU, the same seed gives exactly the same run (see rereading/train.py), for about 11% more time.
if torch.cuda.is_available():
    torch.use_deterministic_algorithms(True)

ARMS = ("self-play", "uniform", "shuffle", "loss_delta")
FRESH, MUTANT, REPLAY = 0, 1, 2   # what each program in the pool is
CLIP = 1.0                        # GUESS: largest gradient length, for both models
EXPERT_WEIGHT = 1.0               # lambda_EI in the paper
SAVE_EVERY = 256                  # rounds, as in the released checkpoints
STATE_FILE = "state.pt"


@dataclass(frozen=True)
class Run:
    arm: str = "self-play"
    seed: int = 0
    rounds: int = 2049            # rounds 0 to 2048; the released checkpoints count rounds the same way
    width: int = 64               # the 99k model: width 64, 1 head, 1 layer
    heads: int = 1
    layers: int = 1
    learning_rate: float = 1e-2   # the learner's, after warmup; inferred from the released weights (see above)
    generator_ratio: float = 0.2  # the generator's learning rate over the learner's; read off their first step
    warmup: int = 50              # rounds; inferred, with the learning rate
    beta: float = 0.01            # GUESS: weight of the pull toward g0 (the KL term)
    adam_beta2: float = 0.95      # GUESS (as in Kim et al.): Adam's averaging of squared gradients, both models
    fresh: int = 512              # GUESS (the split of the pool of 1,536 the authors use)
    mutants: int = 512
    replay: int = 512


def new_model(run, where):
    model = ByteModel(run.width, run.heads, run.layers)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = module.in_features ** -0.5 if isinstance(module, nn.Linear) else 1 / module.embedding_dim
            nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
    return model.to(where)


def learning_rate(round_, run):
    return run.learning_rate * min(1.0, (round_ + 1) / run.warmup)


def build_pool(run, generator, archive, bank, rng, where):
    """This round's programs, what each one is (FRESH, MUTANT or REPLAY), and the replayed ones'
    sampling log-probabilities (nan for the others, and for replayed programs that were mutants)."""
    size = run.fresh + run.mutants + run.replay
    if run.arm == "uniform":
        programs = uniform_programs(size, rng)
        return programs, np.full(size, FRESH), np.full(size, np.nan)
    mutants = archive.mutants(run.mutants, rng) if len(archive) else []
    replayed, replayed_log_probs = bank.draw(run.replay, rng)
    fresh = sample(generator, size - len(mutants) - len(replayed), where)
    kinds = np.array([FRESH] * len(fresh) + [MUTANT] * len(mutants) + [REPLAY] * len(replayed))
    sampled = np.array([np.nan] * (len(fresh) + len(mutants)) + replayed_log_probs)
    return fresh + mutants + replayed, kinds, sampled


def learner_step(learner, optimizer, rows, where):
    """One AdamW step on the mean next-byte loss over all rows (equation 1). Returns that loss."""
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for part in rows.split(reward.CHUNK):
        part = part.to(where)
        with torch.autocast(where.type, dtype=torch.bfloat16):
            logits = learner(part[:, :-1])
        losses = reward.per_row_losses(logits, part)
        (losses.sum() / len(rows)).backward()
        total += losses.sum().item()
    torch.nn.utils.clip_grad_norm_(learner.parameters(), CLIP)
    optimizer.step()
    return total / len(rows)


def generator_step(generator, optimizer, programs, kinds, sampled, rewards, beta, where):
    """One step on the policy gradient plus expert iteration (equations 4 and 5).
    Returns each program's log-probability under the generator before the step."""
    log_g = log_probs(generator, programs, where)
    now = log_g.detach()
    kinds, sampled = torch.from_numpy(kinds).to(where), torch.from_numpy(sampled).float().to(where)
    # Advantage: the reward standardized over the pool, minus beta x log(g / g0).
    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8) - beta * (now - prior_log_probs(programs).to(where))
    # Importance ratio: 1 for fresh programs; for replayed ones, now against when first sampled.
    # Mutants were not sampled, so they stay out of the policy gradient (multiplied by 0: leaving
    # them out by indexing would still pass their nan ratio into the gradient).
    in_gradient = (kinds == FRESH) | ((kinds == REPLAY) & ~sampled.isnan())
    ratio = (now - torch.where(in_gradient & (kinds == REPLAY), sampled, now)).clamp(-20, 20).exp()
    policy = -(in_gradient * ratio * advantage * log_g).sum() / max(int(in_gradient.sum()), 1)
    # Expert iteration: imitate the whole pool, each program weighted by its share of the positive reward.
    positive = rewards.clamp(min=0)
    weights = positive / positive.sum() if positive.sum() > 0 else positive
    expert = -(weights * log_g).sum()
    optimizer.zero_grad(set_to_none=True)
    (policy + EXPERT_WEIGHT * expert).backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), CLIP)
    optimizer.step()
    return now


def save_checkpoints(folder, round_, learner, generator, run):
    """The learner and generator in the released files' format, so score.py reads them."""
    shape = {"d_model": run.width, "n_heads": run.heads, "n_layers": run.layers,
             "max_len": CONTEXT, "vocab_size": VOCABULARY}
    (folder / "config.json").write_text(json.dumps({"learner": shape, "generator": shape}, indent=2))
    torch.save({"round": round_, "learner_state_dict": learner.state_dict()}, folder / f"learner_{round_:04d}.pth")
    if run.arm != "uniform":
        torch.save({"round": round_, "generator_state_dict": generator.state_dict()}, folder / f"generator_{round_:04d}.pth")


def save_safely(save, path):
    """Write to a temporary file, then move it into place, so that a crash while saving (the state
    is about 0.5 GB by round 2048) leaves the previous file whole."""
    temporary = path.with_name(path.name + ".tmp")
    save(temporary)
    os.replace(temporary, path)


def device_rng_state(where):
    if where.type == "cuda":
        return torch.cuda.get_rng_state()
    return torch.mps.get_rng_state() if where.type == "mps" else None


def set_device_rng_state(where, state):
    if where.type == "cuda":
        torch.cuda.set_rng_state(state.cpu())
    elif where.type == "mps":
        torch.mps.set_rng_state(state.cpu())


def train(run, folder, where=None, scores=True):
    """Train one run, saving into folder. Returns its record."""
    assert run.arm in ARMS, f"unknown arm {run.arm}"
    where = where or score.device()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(run.seed)
    rng = np.random.default_rng(run.seed)
    learner, generator = new_model(run, where), new_model(run, where)
    betas = (0.9, run.adam_beta2)
    learner_optimizer = torch.optim.AdamW(learner.parameters(), lr=run.learning_rate, betas=betas, weight_decay=0.0)
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=run.learning_rate * run.generator_ratio,
                                            betas=betas, weight_decay=0.0)
    archive, bank, then = Archive(), ReplayBank(), {}   # then: the learner's weights after each recent round's step
    record = {"run": asdict(run), "rounds": {key: [] for key in (
        "loss", "reward_mean", "reward_spread", "reward_largest", "fresh_symbols", "fresh_kl",
        "wrote_something", "bytes_written", "loop_depth", "niches", "seconds")}, "scores": {}}
    first = 0
    if (folder / STATE_FILE).exists():
        state = torch.load(folder / STATE_FILE, map_location=where, weights_only=False)
        saved = dict(state["record"]["run"], rounds=run.rounds)       # a run may be continued for more rounds
        assert saved == asdict(run), "the saved state is from a different run"
        learner.load_state_dict(state["learner"])
        generator.load_state_dict(state["generator"])
        learner_optimizer.load_state_dict(state["learner_optimizer"])
        generator_optimizer.load_state_dict(state["generator_optimizer"])
        archive.niches, bank.programs, bank.sampled_log_probs = state["niches"], state["bank"], state["bank_log_probs"]
        then, record, first = state["then"], state["record"], state["round"] + 1
        record["run"] = asdict(run)
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        set_device_rng_state(where, state["device_rng"])
    corpora = {name: score.corpus(name) for name in score.CORPORA} if scores else {}
    max_steps, tape_length = machine.limits(run.width, run.layers)

    for round_ in range(first, run.rounds):
        started = time.time()
        for group in learner_optimizer.param_groups:
            group["lr"] = learning_rate(round_, run)
        for group in generator_optimizer.param_groups:
            group["lr"] = learning_rate(round_, run) * run.generator_ratio

        programs, kinds, sampled = build_pool(run, generator, archive, bank, rng, where)
        outputs, written, depths = machine.run(programs, rng.integers(0, 2 ** 63, size=len(programs)),
                                               max_steps, tape_length)
        rows = torch.from_numpy(outputs.astype(np.int64))
        if run.arm == "loss_delta":
            before = reward.sequence_losses(learner, rows, where)
        loss = learner_step(learner, learner_optimizer, rows, where)
        if run.arm in ("self-play", "shuffle"):
            then[round_] = reward.snapshot(learner)
            along = reward.direction(learner, learner_optimizer, then[round_ // 2])
            rewards = reward.alignment(learner, rows, along, where)
        elif run.arm == "loss_delta":
            rewards = (before - reward.sequence_losses(learner, rows, where)).abs()
        if run.arm == "shuffle":
            rewards = rewards[torch.from_numpy(rng.permutation(len(rewards))).to(where)]

        if run.arm != "uniform":
            now = generator_step(generator, generator_optimizer, programs, kinds, sampled, rewards, run.beta, where)
            archive.decay()
            archive.admit(programs, depths, rewards.tolist())
            sampled_now = now.tolist()
            new = [i for i in range(len(programs)) if kinds[i] != REPLAY]
            bank.add([programs[i] for i in new], [sampled_now[i] if kinds[i] == FRESH else np.nan for i in new])
            fresh = torch.from_numpy(kinds == FRESH).to(where)
            fresh_kl = (now - prior_log_probs(programs).to(where))[fresh].mean().item()
        for old in [r for r in then if r < (round_ + 1) // 2]:
            del then[old]

        notes = record["rounds"]
        notes["loss"].append(loss)
        notes["reward_mean"].append(rewards.mean().item() if run.arm != "uniform" else None)
        notes["reward_spread"].append(rewards.std().item() if run.arm != "uniform" else None)
        notes["reward_largest"].append(rewards.max().item() if run.arm != "uniform" else None)
        notes["fresh_symbols"].append(float(np.mean([len(body(p)) for p, k in zip(programs, kinds) if k == FRESH])))
        notes["fresh_kl"].append(fresh_kl if run.arm != "uniform" else 0.0)
        notes["wrote_something"].append(float((written > 0).mean()))
        notes["bytes_written"].append(float(written.mean()))
        notes["loop_depth"].append(float(depths.mean()))
        notes["niches"].append(len(archive.niches))
        notes["seconds"].append(time.time() - started)

        if round_ % SAVE_EVERY == 0 or round_ == run.rounds - 1:
            save_checkpoints(folder, round_, learner, generator, run)
            if scores:
                record["scores"][str(round_)] = {name: score.bits_per_byte(learner, data, where)
                                                 for name, data in corpora.items()}
            state = {"learner": learner.state_dict(), "generator": generator.state_dict(),
                     "learner_optimizer": learner_optimizer.state_dict(),
                     "generator_optimizer": generator_optimizer.state_dict(),
                     "niches": archive.niches, "bank": bank.programs, "bank_log_probs": bank.sampled_log_probs,
                     "then": then, "record": record, "round": round_, "numpy_rng": rng.bit_generator.state,
                     "torch_rng": torch.get_rng_state(), "device_rng": device_rng_state(where)}
            save_safely(lambda path: torch.save(state, path), folder / STATE_FILE)
            save_safely(lambda path: path.write_text(json.dumps(record)), folder / "record.json")
            print(f"round {round_}: loss {loss:.4f}, {np.mean(notes['seconds'][-SAVE_EVERY:]):.2f} s per round, "
                  f"scores {record['scores'].get(str(round_))}", flush=True)
    return record


if __name__ == "__main__":
    settings = {f.name: f.type for f in fields(Run)}
    chosen = {}
    for argument in sys.argv[2:]:
        if argument != "--no-scores":
            name, value = argument.split("=", 1)
            chosen[name] = settings[name](value)
    train(Run(**chosen), sys.argv[1], scores="--no-scores" not in sys.argv)
