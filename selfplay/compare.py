"""Our 99k runs against each other and against the authors' released ones: the pass rule, fixed
before the runs finished.

A run's score on a corpus is the mean of its last three checkpoints, because uniform scores
swing a lot from one checkpoint to the next. Self-play, shuffle and loss_delta: rounds 1536,
1792 and 2048. Uniform: rounds 2560, 2816 and 3072, where it has used as much compute as
self-play has by round 2048 (1,024 programs per round against 1,536).

    gate          our uniform seeds at round 2048 fall inside the released uniform seeds' range
                  on every corpus; if not, our learner differs from theirs and nothing else counts
    reproduction  on audio 8-bit and on DNA, the mean of our self-play seeds is below the mean of
                  our uniform seeds
    ablations     shuffle or loss_delta counts as worse than self-play only if both its seeds are
                  above both self-play seeds, on audio 8-bit (and, second, on DNA)

Run it with:   ../.venv/bin/python compare.py <folder holding the run folders>
"""

import json
import sys
from pathlib import Path

import numpy as np

CORPORA = ("dclm_ranked", "dna", "audio_8bit", "audio_16bit")
LAST = {"uniform": (2560, 2816, 3072)}
DEFAULT_LAST = (1536, 1792, 2048)
ARMS = ("self-play", "uniform", "shuffle", "loss_delta")
RESULTS = Path(__file__).resolve().parent / "results"


def scores(folder, arm):
    """{seed: {corpus: mean over the last three checkpoints}}; seeds with a missing checkpoint are left out."""
    found = {}
    for run in sorted(folder.glob(f"{arm}-[0-9]")):
        saved = json.loads((run / "record.json").read_text())["scores"]
        rounds = LAST.get(arm, DEFAULT_LAST)
        if all(str(r) in saved for r in rounds):
            found[run.name] = {c: float(np.mean([saved[str(r)][c] for r in rounds])) for c in CORPORA}
    return found


def main(folder):
    ours = {arm: scores(folder, arm) for arm in ARMS}
    released = json.loads((RESULTS / "released_99k_round_2048.json").read_text())
    print("Scores, bits per byte (lower is better), each the mean of the last three checkpoints")
    for corpus in CORPORA:
        row = "   ".join(f"{arm} " + "/".join(f"{s[corpus]:.2f}" for s in ours[arm].values()) for arm in ARMS)
        print(f"   {corpus:12s} {row}")

    print("Gate: our uniform seeds at round 2048 against the released uniform seeds")
    theirs = list(released["uniform"].values())
    passed = bool(ours["uniform"])
    for run in sorted(folder.glob("uniform-[0-9]")):
        at_2048 = json.loads((run / "record.json").read_text())["scores"]["2048"]
        for corpus in CORPORA:
            low, high = min(t[corpus] for t in theirs), max(t[corpus] for t in theirs)
            inside = low <= at_2048[corpus] <= high
            passed &= inside
            print(f"   {run.name} {corpus:12s} {at_2048[corpus]:.2f}   theirs {low:.2f}-{high:.2f}   {'inside' if inside else 'OUTSIDE'}")
    print(f"   gate {'passed' if passed else 'FAILED'}")

    print("Reproduction: mean of our self-play seeds against mean of our uniform seeds")
    for corpus in ("audio_8bit", "dna"):
        sp = np.mean([s[corpus] for s in ours["self-play"].values()])
        un = np.mean([s[corpus] for s in ours["uniform"].values()])
        print(f"   {corpus:12s} self-play {sp:.2f}, uniform {un:.2f}: {'self-play better' if sp < un else 'self-play NOT better'}")

    print("Ablations: worse only if both seeds are above both self-play seeds")
    for arm in ("shuffle", "loss_delta"):
        for corpus in ("audio_8bit", "dna"):
            worst_self_play = max(s[corpus] for s in ours["self-play"].values())
            values = [s[corpus] for s in ours[arm].values()]
            worse = len(values) == 2 and min(values) > worst_self_play
            print(f"   {arm:10s} {corpus:12s} " + "/".join(f"{v:.2f}" for v in values) +
                  f" against self-play at most {worst_self_play:.2f}: {'worse' if worse else 'not shown worse'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
