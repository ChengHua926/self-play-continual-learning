"""Before the comparison: do the gradient neighbours on real text mean anything?

The neighbour method's settings are fixed before this look, not chosen from it
(run_comparison.NEIGHBOURS): two sequences are neighbours when their cosine is more than 2
standard deviations above the batch's average, after the batch's average gradient is taken out.
This look is a sanity check, not a judge of the idea; the comparison judges it. It uses the
saved weights of the weight-decay-0.8 run after passes 1, 4, 16 and 32 (run_known_answer.py),
and the training batch size, 64.

    1. Reported, not a rule: do the weights belong to the sequences or to the luck of the batch?
       A fixed pool of POOL training sequences is dealt into random batches ROUNDS times; a
       sequence's average weight over the first half of the rounds against its average over the
       second half. Pure noise gives a correlation near 0. (A high number shows the weights are
       a steady property of each sequence, perhaps only of its words, not that they reflect
       anything learned: an audit found 0.89 even for an untrained model.)
    2. Rule: the weights are not degenerate. At every checkpoint, between 5% and 80% of the
       sequences have no neighbour. If almost none have one, or almost all do, the method is
       close to plain training. If this fails, the comparison is not run: NO-GO for that reason.
    3. Rule: bfloat16 does not change who is a neighbour. Compared with 32-bit arithmetic, at
       most 2% of neighbour pairs change, at every checkpoint. If this fails, the neighbour runs
       use 32-bit (exact=True, about 40 ms more per step) instead.

Run it with:   python look_at_neighbours.py      (from the rereading folder; needs the known-answer
check's saved weights. Don't run run_known_answer.py again for this: see its docstring.)
"""

import numpy as np
import torch

import run_comparison
import run_known_answer
import train
import updates
from model import GPT

BATCH = 64
POOL = 256
ROUNDS = 8
NONE_RANGE = (0.05, 0.80)
FLIPPED = 0.02


def main():
    where = train.device()
    spreads, centred = run_comparison.NEIGHBOURS["spreads"], run_comparison.NEIGHBOURS["centred"]
    fixed = train.windows(np.load(train.TRAINING_FILE)[:run_known_answer.UNIQUE_TOKENS + 1])
    rng = np.random.default_rng(0)
    pool = torch.from_numpy(fixed[rng.choice(len(fixed), POOL, replace=False)]).to(where)
    holds = []
    for passes in run_known_answer.CHECKPOINTS:
        model = GPT(run_known_answer.SIZE).to(where)
        model.load_state_dict(torch.load(run_known_answer.FOLDER / f"wd0.8_after_{passes}_passes.pt", map_location=where))
        model.eval()
        weights = np.zeros((ROUNDS, POOL))
        no_neighbour, fallbacks, flipped = [], 0, []
        for round_ in range(ROUNDS):
            order = rng.permutation(POOL)
            for start in range(0, POOL, BATCH):
                members = order[start:start + BATCH]
                batch = pool[members]
                with torch.no_grad(), train.low_precision(where):
                    features, logits = model(batch[:, :-1])
                inner = updates.gradient_inner_products(features, logits, batch[:, 1:])
                pairs = updates.neighbour_pairs(updates.cosines(inner, centred), spreads)
                counts = pairs.sum(dim=1)
                no_neighbour.append((counts == 0).float().mean().item())
                fallbacks += int(counts.sum() == 0)
                weights[round_, members] = updates.neighbour_weights(updates.cosines(inner, centred), spreads)[0].cpu().numpy()
                if round_ < 2:
                    exact = updates.gradient_inner_products(features, logits, batch[:, 1:], exact=True)
                    exact_pairs = updates.neighbour_pairs(updates.cosines(exact, centred), spreads)
                    either = (pairs | exact_pairs).sum().item()
                    flipped.append((pairs ^ exact_pairs).sum().item() / max(either, 1))
        half = ROUNDS // 2
        stability = np.corrcoef(weights[:half].mean(axis=0), weights[half:].mean(axis=0))[0, 1]
        none, change = float(np.mean(no_neighbour)), float(np.mean(flipped))
        degenerate_ok, precise_ok = NONE_RANGE[0] <= none <= NONE_RANGE[1], change <= FLIPPED
        holds.append((degenerate_ok, precise_ok))
        print(f"After {passes:2d} passes: weights carry over between batches r = {stability:+.2f}; "
              f"no neighbour {none:.0%} ({'ok' if degenerate_ok else 'FAILS'}), "
              f"batches with no neighbour at all {fallbacks} of {ROUNDS * POOL // BATCH}; "
              f"pairs changed by bfloat16 {change:.1%} ({'ok' if precise_ok else 'FAILS'})", flush=True)
    if not all(d for d, _ in holds):
        print("\nRule 2 fails: the weights are degenerate. The comparison is not run: NO-GO for that reason.")
    else:
        print("\nRule 2 holds: the comparison can run, " + ("with bfloat16." if all(p for _, p in holds)
              else "with the neighbour runs in 32-bit (rule 3 fails)."))


if __name__ == "__main__":
    main()
