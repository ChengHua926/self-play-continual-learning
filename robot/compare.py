"""The robot runs against each other: the comparison rule, fixed before any scores were seen.

A run's score is its learner's test 1 loss (scores.py: nats per number on the human demos' steps,
lower is better), the mean of its last three checkpoints (rounds 512, 768 and 1024), because one
checkpoint can be noisy. The runs stop at round 1024, half the paper's budget, to save cost: Cheng
decided this at round 144, before any test 1 score had been seen. Two scores per run: over all
steps, and over the steps where the human holds the nut, where the coverage check found the
baselines' data scarcest.

    the main question   is self-play's world model better than people's way's (the deciding
                        baseline: what people write by hand today)? Better only if both self-play
                        seeds are below both people's-way seeds; worse only if both are above.
    the other arms      self-play against people's way aimed at the nut (more human knowledge),
                        plain random, uniform programs and shuffle, by the same rule.
    a known answer      the learning curves: every run's test 1 loss at its last checkpoint should
                        be below its loss at round 0 (a learner that learns nothing transferable
                        would show a broken setup, not a finding).

Test 2 (plan.py, planning with each final world model) is reported next to it, against the known-
answer curve of planning with the true world.

Run it with:   ../.venv/bin/python compare.py <runs folder>
"""

import json
import sys
from pathlib import Path

import numpy as np

ARMS = ("self-play", "shuffle", "uniform_programs", "plain_random", "peoples_way", "peoples_way_at_nut")
LAST = ("512", "768", "1024")
PARTS = ("all", "holding")


def scores(records):
    """{arm: {part: [score of each seed]}}, each the mean over the last three checkpoints."""
    found = {}
    for record in records:
        arm, saved = record["run"]["arm"], record["scores"]
        if all(r in saved for r in LAST):
            for part in PARTS:
                found.setdefault(arm, {}).setdefault(part, []).append(float(np.mean([saved[r][part] for r in LAST])))
    return found


def verdict(ours, theirs):
    if len(ours) != 2 or len(theirs) != 2:
        return "a seed is missing"
    if max(ours) < min(theirs):
        return "better"
    if min(ours) > max(theirs):
        return "worse"
    return "not shown different"


def main(folder):
    records = json.loads((Path(folder) / "record.json").read_text())
    found = scores(records)
    print("Test 1, nats per number (lower is better), mean of the last three checkpoints, seed 0 / seed 1")
    for part in PARTS:
        print(f"   {part}: " + "   ".join(f"{arm} " + "/".join(f"{s:.3f}" for s in found[arm][part]) for arm in ARMS if arm in found))
    print("Self-play against each other arm: better or worse only if both seeds are beyond both of the other's")
    for other in ARMS[1:]:
        if "self-play" in found and other in found:
            print(f"   against {other:20s} " + "   ".join(f"{part}: {verdict(found['self-play'][part], found[other][part])}" for part in PARTS))
    print("Known answer: every run's test 1 loss at its last checkpoint below its loss at round 0")
    for record in records:
        saved = record["scores"]
        last = max(saved, key=int)
        name = f"{record['run']['arm']}-{record['run']['seed']}"
        print(f"   {name:22s} round 0 {saved['0']['all']:.3f} -> round {last} {saved[last]['all']:.3f}: "
              + ("learned" if saved[last]["all"] < saved["0"]["all"] else "DID NOT LEARN"))


if __name__ == "__main__":
    main(sys.argv[1])
