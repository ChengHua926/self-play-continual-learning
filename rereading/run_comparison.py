"""The go/no-go comparison: does gradient-neighbour weighting beat tuned strong weight decay?

Every run uses the setup of the known-answer check (run_known_answer.py): the 29M model on the
same 20M unique tokens, Kim et al.'s schedule, scored by its held-out loss at the end, on all
the held-out text. Two methods are judged:
    weight decay   plain training with the tuned weight decay, learning rate and number of passes
    neighbours     the same, with each sequence's loss weighted by its gradient neighbours
                   (updates.py), with NEIGHBOURS fixed before any run and checked by
                   look_at_neighbours.py
Every setting is chosen on the tuning seed (100); only then are the two judged on five other
seeds (0 to 4), so neither is tuned on the runs that judge it.

Stage 1, tuning, seed 100:
    - The known-answer check's runs (weight decay 0.1 and 0.8, at 4, 8, 16 and 32 passes) are
      reused if one of them, run again under this code, ends within SAME_RERUN of its saved
      result; otherwise all are run again. (They ran before train.py switched the GPU to its
      deterministic mode, so a rerun is not expected to match, and they will be run again.)
    - Weight decay 1.6 and 3.2 at 16 and 32 passes; if 3.2 is best, also 6.4 (the top of Kim et
      al.'s grid). The tuned weight decay and its passes P are the pair that ends lowest.
    - One run at learning rate 6e-3 (Kim et al.'s best, 3e-3, was the top of their grid): if it
      ends lower, every later run uses 6e-3.
    - Neighbours get P or 2P passes, whichever ends lower (weighting may slow learning). No run
      has more than MAX_PASSES passes.
Stage 2, judging: each method once on each of seeds 0 to 4.

The rule, fixed before any of these runs:
    GO if neighbours end below tuned weight decay on all five judged seeds, and their average
    gain is at least MIN_GAIN. Otherwise NO-GO.
With no real difference, winning on all five seeds happens by luck 1 time in 32. MIN_GAIN is a
practical floor: Kim et al. found runs on different seeds differ by about 0.008.
Only if GO, two controls run on the same five seeds, with the neighbours' settings:
    shuffled neighbours   the same weights handed out in a random order. GO stands only if
                          neighbours also end below it on all five seeds: otherwise what
                          helped was reweighting, not neighbours.
    batch agreement       weights from agreement with the whole batch instead of a group.
                          Reported, not a rule: it says whether finding the groups mattered.

Cost, measured on the A100 before stage 1: a plain step takes 87 ms and a neighbours step 178
ms, so a pass over the 20M tokens takes about 1 minute plain and 2 minutes with neighbours;
deterministic mode (train.py) adds about 11%.
With P = 16, stage 1 takes about 3.5 to 4.5 hours and stage 2 about 4 to 6.5, together about
$15-22 at $1.99 an hour; with P = 32, up to about 12 hours, $24. The controls, if they run,
add about 5 to 10 hours.

Run it with:   python run_comparison.py      (from the rereading folder, on the GPU machine)
Each run is saved to results/comparison/ as it ends, with its settings and code version; running
it again skips the runs already saved with the same settings and code.
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import run_known_answer
import train
from run_known_answer import fingerprint, save
from train import Run

FOLDER = Path("results/comparison")
TUNING_SEED = 100
JUDGED_SEEDS = (0, 1, 2, 3, 4)
MORE_DECAYS = {1.6: (16, 32), 3.2: (16, 32)}
HIGHEST_DECAY = {6.4: (16, 32)}
FASTER_LEARNING_RATE = 6e-3
MAX_PASSES = 32
NEIGHBOURS = {"spreads": 2.0, "centred": True, "exact": False}  # exact becomes True only if look_at_neighbours' rule 3 fails
MIN_GAIN = 0.01      # nats per token
SAME_RERUN = 0.005   # nats per token


def run_file(run):
    return FOLDER / f"{run.method.replace(' ', '_')}_wd{run.weight_decay}_lr{run.learning_rate}_passes{run.passes}_seed{run.seed}.json"


def record_of(run):
    """The run's record: from its saved file if the settings and code match, else trained now and saved."""
    path, settings = run_file(run), fingerprint(run)
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("fingerprint") == settings:
            return record
    FOLDER.mkdir(parents=True, exist_ok=True)  # before training: the run saves its final weights there
    record = train.train(run, final_weights_file=str(path.with_suffix(".pt")))
    record["fingerprint"] = settings
    save(record, path)
    print(f"{run.method}, weight decay {run.weight_decay}, learning rate {run.learning_rate}, {run.passes} passes, "
          f"seed {run.seed}: {record['final_held_out']:.4f}  ({record['seconds'] / 60:.0f} minutes)", flush=True)
    return record


def end(record):
    """A judged run's held-out loss at the end. A diverged judged run stops everything."""
    assert math.isfinite(record["final_held_out"]), f"a run diverged: {record['run']}"
    return record["final_held_out"]


def tuning_end(record):
    """A tuning run's held-out loss at the end; a diverged one counts as infinitely bad, so it is never chosen."""
    return record["final_held_out"] if math.isfinite(record["final_held_out"]) else math.inf


def known_answer_ends(tuning):
    """The known-answer check's end losses, reused if one rerun under this code matches; else all run again."""
    runs = {(d, p): replace(tuning, weight_decay=d, passes=p)
            for d in run_known_answer.WEIGHT_DECAYS for p in run_known_answer.PASSES}
    records = {key: json.loads(run_known_answer.run_file(run).read_text()) for key, run in runs.items()}
    shared = ("size", "unique_tokens", "passes", "method", "weight_decay", "learning_rate", "sequences", "seed")
    for key, run in runs.items():
        assert all(records[key]["run"][f] == getattr(run, f) for f in shared), f"the saved known-answer run {key} differs"
    again = record_of(runs[0.1, 4])
    same = abs(tuning_end(again) - tuning_end(records[0.1, 4])) <= SAME_RERUN
    print("Weight decay 0.1, 4 passes, saved and run again under this code (held-out loss along the way, then at the end):")
    for name, record in (("saved", records[0.1, 4]), ("again", again)):
        print(f"  {name}: " + " ".join(f"{loss:.3f}" for loss in record["held_out"]) + f" | {record['final_held_out']:.4f}")
    print("  The known-answer runs are reused." if same else "  Too different, so all are run again.", flush=True)
    if same:
        return {key: tuning_end(record) for key, record in records.items()}
    return {key: tuning_end(record_of(run)) for key, run in runs.items()}


def tune():
    """Stage 1, on the tuning seed: the tuned weight-decay run and the neighbour run to judge."""
    tuning = replace(Run(size=run_known_answer.SIZE, unique_tokens=run_known_answer.UNIQUE_TOKENS), seed=TUNING_SEED)
    ends = known_answer_ends(tuning)
    for decays in (MORE_DECAYS, HIGHEST_DECAY):
        if decays is HIGHEST_DECAY and min(ends, key=ends.get)[0] != max(MORE_DECAYS):
            break
        ends |= {(d, p): tuning_end(record_of(replace(tuning, weight_decay=d, passes=p)))
                 for d, ps in decays.items() for p in ps}
    decay, passes = min(ends, key=ends.get)
    print("\nTuning seed, held-out loss at the end by weight decay and passes:")
    for (d, p), loss in sorted(ends.items()):
        print(f"  weight decay {d:>4}, {p:2d} passes: {loss:.4f}{'   <- tuned' if (d, p) == (decay, passes) else ''}")
    faster = tuning_end(record_of(replace(tuning, weight_decay=decay, passes=passes, learning_rate=FASTER_LEARNING_RATE)))
    rate = FASTER_LEARNING_RATE if faster < ends[decay, passes] else tuning.learning_rate
    print(f"  learning rate {FASTER_LEARNING_RATE} at the tuned pair: {faster:.4f}; learning rate used: {rate}")
    weight_decay = replace(tuning, weight_decay=decay, passes=passes, learning_rate=rate)
    neighbours = replace(weight_decay, method="neighbours", **NEIGHBOURS)
    options = sorted({passes, min(2 * passes, MAX_PASSES)})
    neighbour_records = {p: record_of(replace(neighbours, passes=p)) for p in options}
    neighbour_ends = {p: tuning_end(record) for p, record in neighbour_records.items()}
    neighbours = replace(neighbours, passes=min(neighbour_ends, key=neighbour_ends.get))
    for p, record in neighbour_records.items():
        print(f"  neighbours, {p} passes: {neighbour_ends[p]:.4f}; weights along the way: at 0 "
              + " ".join(f"{z:.0%}" for z in record["zero_weights"][::4]) + ", effective batch "
              + " ".join(f"{e:.0f}" for e in record["effective_batch"][::4]))
    plain = min(((p, ends[0.1, p]) for p in run_known_answer.PASSES), key=lambda pair: pair[1])
    return weight_decay, neighbours, plain


def judged(run):
    """Its records on the judged seeds."""
    return [record_of(replace(run, seed=seed)) for seed in JUDGED_SEEDS]


def main():
    replace(Run(), method="neighbours", **NEIGHBOURS)  # a misspelt setting fails now, not hours in
    weight_decay, neighbours, plain = tune()
    print(f"\nSettings from the tuning seed: weight decay {weight_decay.weight_decay}, learning rate "
          f"{weight_decay.learning_rate}; {weight_decay.passes} passes for weight decay, {neighbours.passes} for neighbours")
    decay_records, neighbour_records = judged(weight_decay), judged(neighbours)
    decay_ends, neighbour_ends = [end(r) for r in decay_records], [end(r) for r in neighbour_records]
    gains = [d - n for d, n in zip(decay_ends, neighbour_ends)]
    go = all(g > 0 for g in gains) and sum(gains) / len(gains) >= MIN_GAIN

    print("\nHeld-out loss at the end, seeds " + ", ".join(map(str, JUDGED_SEEDS)) + "; in brackets, the gap to the")
    print("loss on the training sample, which grows with memorizing:")
    for name, records in (("weight decay", decay_records), ("neighbours", neighbour_records)):
        print(f"  {name:>12s}: " + "  ".join(f"{end(r):.4f} ({end(r) - r['final_training_sample']:.2f})" for r in records))
    print(f"  (plain training, weight decay 0.1, at its best {plain[0]} passes on the tuning seed: {plain[1]:.4f})")
    print(f"\nGains of neighbours over weight decay: " + ", ".join(f"{g:+.4f}" for g in gains)
          + f"; average {sum(gains) / len(gains):+.4f} (rule: above 0 on every seed, and at least {MIN_GAIN} on average)")
    if not go:
        print("Verdict, by the rule fixed before the runs: NO-GO")
        return
    shuffled = [end(r) for r in judged(replace(neighbours, method="shuffled neighbours"))]
    agreement = [end(r) for r in judged(replace(neighbours, method="batch agreement"))]
    beats_shuffled = all(n < s for n, s in zip(neighbour_ends, shuffled))
    print("Shuffled neighbours: " + "  ".join(f"{s:.4f}" for s in shuffled)
          + f"; neighbours below it on every seed: {'yes' if beats_shuffled else 'no'}")
    print("Not a rule, batch agreement: " + "  ".join(f"{a:.4f}" for a in agreement))
    print(f"Verdict, by the rule fixed before the runs: {'GO' if beats_shuffled else 'NO-GO (reweighting, not neighbours)'}")


if __name__ == "__main__":
    main()
