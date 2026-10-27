"""Read the map: print where choosing pays under each guard, and draw one figure.

Run it with:   python3 report_map.py            (the plain worlds)
         or:   python3 report_map.py narrow     (the narrow worlds)

The main number is the change in true error on the learnable objects (all but the toys)
at the end of the run, compared with even pushing in the same worlds and under the same
guard. Negative means the learner did better than spreading its pushes evenly. Next to
it: in how many of the runs it did better.

Then the share of pushes each learner spent on toys, and a check of the guards
themselves: how much toy noise was memorized, and how well the learnable objects were
learned, by even pushing, which chooses nothing.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from report_step1 import FAINT_INK, GRID, INK, PAPER, SOFT_INK, tidy

END_ROUNDS = 5  # "the end of the run" is the average over the last this-many rounds

# The table is orange and dashed. Random worlds are blue, light to dark from small to big.
WORLD_STYLES = {"table": ("#eb6834", "--"), "6": ("#9cc3ee", "-"), "24": ("#2a78d6", "-"), "96": ("#123f7a", "-")}


def results_file(world_set):
    return "results/map.npz" if world_set == "plain" else f"results/map_{world_set}.npz"


def main(world_set="plain"):
    saved = np.load(results_file(world_set))
    learners, training = list(saved["learners"]), list(saved["training"])
    worlds, guards = list(saved["worlds"]), list(saved["guards"])
    even = learners.index("even pushing")
    changes = []

    for g, guard in enumerate(guards):
        print(f"======== {world_set} worlds, guard: {guard} ========\n")
        end = saved["learnable"][:, g, ..., -END_ROUNDS:].mean(axis=-1)  # [world, training setting, learner, run]
        change = end / end[:, :, even:even + 1] - 1                     # against even pushing, run by run
        changes.append(change)
        print_changes(change, learners, training, worlds, even)
        print_toy_share(saved["toy_share"][:, g], learners, training, worlds)

    print_guard_check(saved, learners, training, worlds, guards, even)
    draw_map(changes, guards, learners, training, worlds, even, world_set)


def world_name(world):
    narrow = "narrow " if world.startswith("narrow ") else ""
    base = world.removeprefix("narrow ")
    return f"{narrow}table (steps 1, 2)" if base == "table" else f"{narrow}{base} objects (random)"


def print_changes(change, learners, training, worlds, even):
    print("Change in true error on the learnable objects at the end of the run, against even pushing.")
    print("Negative is better. In brackets: in how many runs the learner beat even pushing.\n")
    for w, world in enumerate(worlds):
        print(f"{world_name(world):27s}" + "".join(f"{f'{steps} steps':>13s}" for steps in training))
        for l, learner in enumerate(learners):
            if l == even:
                continue
            cells = [change[w, t, l] for t in range(len(training))]
            print(f"  {learner:25s}" + "".join(f"{c.mean():+7.0%} ({(c < 0).sum()}/{len(c)})" for c in cells))
        print()


def print_toy_share(toy_share, learners, training, worlds):
    print("Share of all pushes that went to toys (toys are 1 in 6 of the objects).\n")
    for w, world in enumerate(worlds):
        print(f"{world_name(world):27s}" + "".join(f"{f'{steps} steps':>11s}" for steps in training))
        for l, learner in enumerate(learners):
            print(f"  {learner:25s}" + "".join(f"{toy_share[w, t, l].mean():11.0%}" for t in range(len(training))))
        print()


def print_guard_check(saved, learners, training, worlds, guards, even):
    """What each guard did, shown for even pushing, which chooses nothing."""
    print("======== The guards themselves, for even pushing ========\n")
    stopping = guards.index("early stopping")
    print("Training steps actually taken per round, on average, with early stopping:")
    print(f"{'':27s}" + "".join(f"{f'{steps} steps':>11s}" for steps in training))
    for w, world in enumerate(worlds):
        print(f"  {world_name(world):25s}" + "".join(
            f"{saved['steps_taken'][w, stopping, t, even].mean():11.0f}" for t in range(len(training))))
    for line, what in (("toy_error", "True error of the toys at the end (memorized noise)"),
                       ("learnable", "True error of the learnable objects at the end")):
        print(f"\n{what}, by guard: " + " / ".join(guards))
        print(f"{'':27s}" + "".join(f"{f'{steps} steps':>24s}" for steps in training))
        for w, world in enumerate(worlds):
            end = saved[line][w, :, :, even, :, -END_ROUNDS:].mean(axis=(-2, -1))  # [guard, training setting]
            print(f"  {world_name(world):25s}" + "".join(
                f"{' / '.join(f'{end[g, t]:.4f}' for g in range(len(guards))):>24s}" for t in range(len(training))))
    print()


def draw_map(changes, guards, learners, training, worlds, even, world_set):
    """One row per guard, one panel per label, one line per world."""
    labels = [learner for learner in learners if learner != "even pushing"]
    # Each column gets its own scale: raw error is so far off that a shared scale would flatten the rest.
    figure, panels = plt.subplots(len(guards), len(labels), figsize=(16, 3.1 * len(guards) + 0.8),
                                  sharey="col", facecolor=PAPER)
    for g, guard in enumerate(guards):
        for panel, label in zip(panels[g], labels):
            l = learners.index(label)
            for w, world in enumerate(worlds):
                colour, style = WORLD_STYLES[world.removeprefix("narrow ")]
                panel.plot(training, changes[g][w, :, l].mean(axis=-1), color=colour, linestyle=style, linewidth=2,
                           marker="o", markersize=4, label=world_name(world))
            panel.axhline(0, color=FAINT_INK, linewidth=1)
            panel.set_xscale("log")
            panel.set_xticks(training, [str(steps) for steps in training])
            panel.minorticks_off()
            panel.yaxis.set_major_formatter(PercentFormatter(1))
            panel.set_title(f"{label}, guard: {guard}", loc="left", fontsize=9, color=INK)
            tidy(panel)
            panel.grid(axis="y", color=GRID, linewidth=1)
    for panel in panels[-1]:
        panel.set_xlabel("training steps per round", color=SOFT_INK, fontsize=9)
    figure.suptitle(
        f"{world_set.capitalize()} worlds: change in true error on the learnable objects, against even pushing "
        "(below zero: choosing paid)",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    figure.legend(*panels[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.965),
                  ncol=len(worlds), frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(results_file(world_set).replace(".npz", ".png"), dpi=160)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plain")
