"""Read step A: does the ranking of the labels hold with a fair budget, and do the critics
forecast their labels?

Run it with:   python3 report_step_a.py      (after run_step_a.py has written the results)

It prints three parts and draws one figure:

    1. The change in true error on the learnable objects (all but the toys) at the end of
       the run, against even pushing with the same budget. Negative means the learner did
       better than spreading its pushes evenly. Next to it: in how many runs it did better.
       Shown for the old doubled budget and the fair one, on the same worlds. Differences
       of a few percent are within the spread from run to run: the order of labels that
       close together means nothing.
    2. The share of pushes that went to toys.
    3. The critic check (see run_step2.check_critic): how well each critic put random
       pushes in the same order as the grades they then got, next to the object average,
       a forecaster that knows only which object a push is for. 1 is the same order, 0 is
       no relation. Each round it is scored on about 32 pushes (16 for repeat agreement,
       and for every label under the doubled budget), so a single round says little; the
       averages over 70 rounds and 10 runs are what to read.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from report_map import END_ROUNDS, world_name
from report_step1 import FAINT_INK, GRID, INK, PAPER, SOFT_INK, tidy
from run_step2 import CRITIC_CHECKS

RESULTS_FILE = "results/step_a.npz"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LIMIT = 0.3  # the figure shows changes from -30% to +30%; a point further out is drawn at the edge, with its value


def main():
    saved = np.load(RESULTS_FILE)
    budgets, worlds, learners = list(saved["budgets"]), list(saved["worlds"]), list(saved["learners"])
    even = learners.index("even pushing")
    labels = [learner for learner in learners if learner != "even pushing"]

    end = saved["learnable"][..., -END_ROUNDS:].mean(axis=-1)  # [budget, world, learner, run]
    change = end / end[:, :, even:even + 1] - 1                # against even pushing with the same budget, run by run
    # The critic check, averaged over the rounds in which it was made: {name: [budget, world, learner, run]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # even pushing has no critic, so its lines are empty
        check = {name: np.nanmean(saved[name], axis=-1) for name in CRITIC_CHECKS}

    print_changes(change, end, budgets, worlds, learners, labels)
    print_toy_share(saved["toy_share"], budgets, worlds, learners)
    print_critic_check(check, budgets, worlds, learners, labels)
    draw(change, check, budgets, worlds, learners, labels)


def print_changes(change, end, budgets, worlds, learners, labels):
    print("1. Change in true error on the learnable objects at the end of the run, against even pushing")
    print("   with the same budget. Negative is better. In brackets: in how many runs the learner beat even pushing.\n")
    print(f"{'':27s}" + "".join(f"{budget + ' budget':>17s}" for budget in budgets))
    even, doubled, fair = learners.index("even pushing"), budgets.index("doubled"), budgets.index("fair")
    for w, world in enumerate(worlds):
        print(world_name(world))
        for label in labels:
            cells = [change[b, w, learners.index(label)] for b in range(len(budgets))]
            print(f"  {label:25s}" + "".join(f"{c.mean():+9.0%} ({(c < 0).sum():2d}/{len(c)})" for c in cells))
        # What even pushing itself gained from making new pushes instead of copies.
        gain = end[fair, w, even] / end[doubled, w, even] - 1
        print(f"  even pushing itself, fair against doubled budget: {gain.mean():+.0%} ({(gain < 0).sum()}/{len(gain)} runs better)\n")


def print_toy_share(toy_share, budgets, worlds, learners):
    print("2. Share of all pushes that went to toys (toys are 1 in 6 of the objects), doubled / fair budget.\n")
    print(f"{'':27s}" + "".join(f"{world:>14s}" for world in worlds))
    for l, learner in enumerate(learners):
        cells = [" / ".join(f"{toy_share[b, w, l].mean():.0%}" for b in range(len(budgets))) for w in range(len(worlds))]
        print(f"  {learner:25s}" + "".join(f"{cell:>14s}" for cell in cells))
    print()


def print_critic_check(check, budgets, worlds, learners, labels):
    print("3. The critic check: how well the forecasts put random pushes in the same order as the grades")
    print("   they then got (1 = same order, 0 = no relation), averaged over the rounds and runs.")
    print("   critic: the learner's critic. object average: knows only which object a push is for.\n")
    for b, budget in enumerate(budgets):
        print(f"{budget + ' budget':27s}{'all pushes':>24s}{'without the toys':>24s}")
        print(f"{'':27s}" + f"{'critic':>12s}{'object avg':>12s}" * 2)
        for w, world in enumerate(worlds):
            print(f"  {world_name(world)}")
            for label in labels:
                l = learners.index(label)
                print(f"    {label:23s}" + "".join(f"{check[name][b, w, l].mean():+12.2f}" for name in CRITIC_CHECKS))
        print()


def draw(change, check, budgets, worlds, learners, labels):
    """Top row: the change against even pushing, doubled and fair budget. Bottom row: the critic check."""
    figure, panels = plt.subplots(2, len(worlds), figsize=(15, 6.4), sharey=True, facecolor=PAPER)
    rows = np.arange(len(labels))[::-1]  # the first label at the top
    doubled, fair = budgets.index("doubled"), budgets.index("fair")
    for w, world in enumerate(worlds):
        top, bottom = panels[0, w], panels[1, w]
        old = np.array([change[doubled, w, learners.index(label)].mean() for label in labels])
        new = np.array([change[fair, w, learners.index(label)].mean() for label in labels])
        for row, before, after in zip(rows, old, new):
            if max(abs(before), abs(after)) > LIMIT:
                top.text(LIMIT, row - 0.3, f"off the chart: {before:+.0%}, {after:+.0%}", ha="right", va="top",
                         fontsize=8, color=SOFT_INK)
        old, new = np.clip(old, -LIMIT, LIMIT), np.clip(new, -LIMIT, LIMIT)
        top.hlines(rows, old, new, color=FAINT_INK, linewidth=1.5)
        top.scatter(old, rows, s=40, facecolors=PAPER, edgecolors=SOFT_INK, linewidths=1.5, zorder=3, label="doubled budget")
        top.scatter(new, rows, s=40, color=BLUE, zorder=3, label="fair budget")
        top.axvline(0, color=FAINT_INK, linewidth=1)
        top.set_xlim(-LIMIT * 1.05, LIMIT * 1.05)
        top.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        top.set_title(short_name(world), loc="left", fontsize=10, color=INK)

        critic = [check["critic"][fair, w, learners.index(label)].mean() for label in labels]
        average = [check["object_average"][fair, w, learners.index(label)].mean() for label in labels]
        bottom.hlines(rows, average, critic, color=FAINT_INK, linewidth=1.5)
        bottom.scatter(average, rows, s=40, facecolors=PAPER, edgecolors=ORANGE, linewidths=1.5, zorder=3,
                       label="object average (knows only the object)")
        bottom.scatter(critic, rows, s=40, color=BLUE, zorder=3, label="the learner's critic")
        bottom.axvline(0, color=FAINT_INK, linewidth=1)

        for panel in (top, bottom):
            panel.set_yticks(rows, labels)
            tidy(panel)
            panel.grid(axis="x", color=GRID, linewidth=1)
    panels[0, 0].set_ylabel("change in true error against\neven pushing (left is better)", color=SOFT_INK, fontsize=9)
    panels[1, 0].set_ylabel("critic check, fair budget\n(1 = same order as the grades)", color=SOFT_INK, fontsize=9)
    panels[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, frameon=False, fontsize=9)
    panels[1, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(RESULTS_FILE.replace(".npz", ".png"), dpi=160)


def short_name(world):
    """A world's name, short enough for the title of a small panel."""
    narrow = world.startswith("narrow ")
    return f"{world.removeprefix('narrow ')} objects" + (", narrow" if narrow else "")


if __name__ == "__main__":
    main()
