"""Read the game comparison (run_game_comparison.py).

Run it with:   .venv/bin/python report_game_comparison.py     (after run_game_comparison.py)

Knowledge (test_set.py) is the share of the game's movement the world model predicts, room by
room, every room counting the same: 0 is no better than guessing "nothing moves", 1 is perfect.
Each run is read by its knowledge averaged over the whole run (every 100,000 frames from the
start to 8 million); "the end" is the average of the last 5 measurements, the last 500,000 frames.

For every learner, against uniform choice ("not choosing"), run k against run k:
    knowledge over the whole run, at the end, and early (after 2 million frames)
    frames saved: how many fewer frames it needs to reach uniform choice's end knowledge, on
        the knowledge averaged over its runs and smoothed over 5 measurements
    in how many of its 10 runs its whole-run knowledge is above uniform choice's
    rooms reached by the end, and when the first room was left (median over the runs that left)
Then the verdict by the rules fixed before the run (run_game_comparison.py), and four figures:
    game_comparison.png         one row per learner: its 10 runs' whole-run knowledge as small dots,
                                their average as a big one
    game_comparison_curves.png  knowledge against frames: the baselines and the best of each family
    game_comparison_rooms.png   rooms reached against frames, the same learners
    game_comparison_map.png     which test rooms each learner reached, in how many of its 10 runs
"""

from math import comb

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from report_step1 import FAINT_INK, GRID, INK, PAPER, SOFT_INK, tidy

RESULTS_FILE = "results/game_comparison.npz"
END = 5            # "the end" is the average of the last 5 measurements
EARLY = 2_000_000  # frames
WINS_NEEDED = 7    # of the 10 runs
FLAG_BELOW = 0.05  # flagged if its average whole-run knowledge is more than 5% below uniform choice's
FAMILIES = {
    "baselines": ("uniform choice", "random play", "do-nothing critic"),
    "novelty": ("cell counts", "feature novelty", "feature novelty x steepness", "random-network novelty",
                "random-network novelty x steepness"),
    "uncertainty": ("ensemble disagreement", "disagreement x steepness", "pinned weights", "steepness"),
    "critics": ("raw error critic", "progress over 10 rounds critic", "new places found critic",
                "disagreement found critic"),
}
FAMILY_COLORS = {"baselines": INK, "novelty": "#1baf7a", "uncertainty": "#2a78d6", "critics": "#eb6834"}


def main():
    saved = np.load(RESULTS_FILE)
    names = list(saved["learners"])
    uniform = names.index("uniform choice")
    room_knowledge = np.maximum(1 - saved["room_errors"] / saved["no_change"], 0)  # [learner, run, measurement, room]
    assert np.isfinite(room_knowledge).all(), "a world model diverged: its knowledge is not a number"
    knowledge = room_knowledge.mean(axis=-1)                                        # [learner, run, measurement]
    frames = saved["frames"][uniform, 0]
    whole = knowledge.mean(axis=-1)                                                 # [learner, run]
    end = knowledge[..., -END:].mean(axis=-1)
    early = knowledge[..., np.argmin(np.abs(frames - EARLY))]
    wins = (whole > whole[uniform]).sum(axis=-1)
    rooms, rooms_end = saved["rooms"], saved["rooms"][..., -1]
    saved_frames = frames_saved(knowledge, frames, uniform)

    order = [names.index(name) for family in FAMILIES.values() for name in family]
    print("Knowledge over the whole run, at the end and early (2 million frames); frames saved against uniform\n"
          "choice; runs of 10 above uniform choice over the whole run; rooms reached by the end; when the first\n"
          "room was left (median, and how many of the 10 runs left it).\n")
    print(f"{'':36s}{'whole run':>10s}{'end':>7s}{'early':>7s}{'frames saved':>14s}{'above uniform':>15s}"
          f"{'rooms':>7s}{'left room 1':>13s}")
    for family, members in FAMILIES.items():
        print(family)
        for name in members:
            l = names.index(name)
            above = "-" if l == uniform else f"{wins[l]}/10"
            saving = "never" if np.isnan(saved_frames[l]) else f"{saved_frames[l]:+.0%}"
            left = [frames[np.flatnonzero(run_rooms > 1)[0]] / 1e6 for run_rooms in rooms[l] if (run_rooms > 1).any()]
            left_text = f"{np.median(left):.1f}M ({len(left)})" if left else "never"
            print(f"  {name:34s}{whole[l].mean():10.1%}{end[l].mean():7.1%}{early[l].mean():7.1%}{saving:>14s}"
                  f"{above:>15s}{rooms_end[l].mean():7.1f}{left_text:>13s}")

    works = [names[l] for l in order if l != uniform and wins[l] >= WINS_NEEDED]
    flagged = [names[l] for l in order if whole[l].mean() < (1 - FLAG_BELOW) * whole[uniform].mean()]
    by_luck = sum(comb(10, k) for k in range(WINS_NEEDED, 11)) / 2**10
    print("\nVerdict, by the rules fixed before the run")
    print(f"  Works (whole-run knowledge above uniform choice's in at least {WINS_NEEDED} of 10 runs): "
          f"{'; '.join(works) or 'none'}.")
    print(f"  Flagged (average whole-run knowledge more than {FLAG_BELOW:.0%} below uniform choice's): "
          f"{'; '.join(flagged) or 'none'}.")
    print(f"  Luck: with no real difference, {WINS_NEEDED} or more of 10 runs happen {by_luck:.0%} of the time, so among "
          f"{len(names) - 1} learners about {(len(names) - 1) * by_luck:.1f} would work by luck alone.")

    draw_runs(names, whole, uniform)
    best = [max((names.index(name) for name in members), key=lambda l: whole[l].mean())
            for family, members in FAMILIES.items() if family != "baselines"]
    shown = [names.index(name) for name in FAMILIES["baselines"]] + best
    draw_lines(names, knowledge, frames, shown, "knowledge of the game (higher is better)", "_curves")
    draw_lines(names, saved["rooms"].astype(float), frames, shown, "rooms reached", "_rooms")
    draw_map(names, saved["rooms_reached"][:, :, -1], saved["rooms_tested"], order)


def frames_saved(knowledge, frames, reference):
    """How many fewer frames each learner needs to reach the reference's end knowledge (a gap for never).

    Each learner's knowledge is averaged over its runs and smoothed: the window ending at
    measurement i is the average of measurements i - 4 to i. The first window at or above the
    reference's end knowledge is where it gets there.
    """
    smooth = sliding_window_view(knowledge.mean(axis=1), END, axis=-1).mean(axis=-1)  # [learner, window]
    window_end = frames[END - 1:]
    target = smooth[reference, -1]
    high_enough = smooth >= target
    needed = np.where(high_enough.any(axis=-1), window_end[high_enough.argmax(axis=-1)], np.nan)
    return 1 - needed / needed[reference]


def family_of(name):
    return next(family for family, members in FAMILIES.items() if name in members)


def draw_runs(names, whole, uniform):
    """One row per learner: its 10 runs' whole-run knowledge as small dots, the average as a big one."""
    rows = [name for members in FAMILIES.values() for name in members]
    figure, panel = plt.subplots(figsize=(9, 7), facecolor=PAPER)
    y = -np.arange(len(rows))
    for row, name in zip(y, rows):
        l, color = names.index(name), FAMILY_COLORS[family_of(name)]
        panel.scatter(whole[l], np.full(len(whole[l]), row), s=12, color=color, alpha=0.45, linewidths=0)
        panel.scatter(whole[l].mean(), row, s=70, color=color, zorder=3)
    panel.axvline(whole[uniform].mean(), color=FAINT_INK, linewidth=1.2, linestyle="--")
    panel.set_yticks(y, rows)
    panel.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    panel.set_xlabel("knowledge of the game, averaged over the whole run (dashed: uniform choice's average)",
                     color=SOFT_INK, fontsize=9)
    tidy(panel)
    panel.grid(axis="x", color=GRID, linewidth=1)
    figure.tight_layout()
    figure.savefig(RESULTS_FILE.replace(".npz", ".png"), dpi=160)


def draw_lines(names, line, frames, shown, label, suffix):
    """A line over frames for each shown learner: the average over runs."""
    figure, panel = plt.subplots(figsize=(9, 4.6), facecolor=PAPER)
    styles = {"uniform choice": "-", "random play": ":", "do-nothing critic": "--"}
    for l in shown:
        name = names[l]
        panel.plot(frames / 1e6, line[l].mean(axis=0), color=FAMILY_COLORS[family_of(name)],
                   linestyle=styles.get(name, "-"), linewidth=2, label=name)
    tidy(panel)
    panel.grid(axis="y", color=GRID, linewidth=1)
    panel.set_xlabel("frames played (millions)", color=SOFT_INK, fontsize=9)
    panel.set_ylabel(label, color=SOFT_INK, fontsize=9)
    if "knowledge" in label:
        panel.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    panel.legend(loc="upper left", frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(RESULTS_FILE.replace(".npz", f"{suffix}.png"), dpi=160)


def draw_map(names, reached, rooms_tested, order):
    """Which test rooms each learner reached, in how many of its 10 runs."""
    share = reached.mean(axis=1)[order]  # [learner, test room]
    figure, panel = plt.subplots(figsize=(8, 7), facecolor=PAPER)
    panel.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            panel.text(j, i, f"{share[i, j] * 10:.0f}", ha="center", va="center", fontsize=7,
                       color="white" if share[i, j] > 0.6 else INK)
    panel.set_xticks(range(len(rooms_tested)), [str(room) for room in rooms_tested])
    panel.set_yticks(range(len(order)), [names[l] for l in order])
    panel.set_xlabel("test room (number: runs of 10 that reached it)", color=SOFT_INK, fontsize=9)
    panel.tick_params(colors=SOFT_INK, labelsize=8, length=0)
    figure.tight_layout()
    figure.savefig(RESULTS_FILE.replace(".npz", "_map.png"), dpi=160)


if __name__ == "__main__":
    main()
