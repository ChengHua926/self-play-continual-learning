"""Read the toy comparison: which ways of choosing pushes help a learner, end to end?

Run it with:   python3 report_comparison.py      (after run_comparison.py has written the results)

The error. After every round, each learner's true error on the learnable objects (every
object on the table but the toys), measured against the world's rules. "End error" is its
average over the last 5 rounds.

Pushes saved, the headline number. Not choosing ends the run at some error. How many fewer
pushes does a learner need to get down to that same error? Every learner makes 256 actions
per round, so pushes are counted in rounds. "Saves 20%" means it gets there with a fifth
fewer pushes; "never" means it does not get there in 70 rounds.

How it is found: each learner's error is averaged over its 10 runs and smoothed with the
same 5-round average as the end error. Late objects arrive in rounds 20 and 40, and until
they are all on the table, the error of the others can be lower than any end error without
meaning anything. So the search starts with the 5 rounds from round 40 on, and a learner
can show at most about 35% saved. If 3 or more learners sit at that cap in a narrow world,
the report counts only the objects on the table from the first round instead, where the
search can start at once (decided before the run).

The reading rules, fixed before the run:

    A learner works if, in both narrow worlds, it ends with lower true error than the
    do-nothing critic in at least 7 of the 10 runs. (A run of every learner has the same
    world, the same starting network and the same test pushes, so runs compare one to one.)

    A bet wins if it ends with lower true error than the best of the bar to beat in at
    least 7 of the 10 runs, in both narrow worlds. The best of the bar to beat is the one
    with the lowest end error against not choosing, averaged over the two narrow worlds.

    Plain worlds are a check only: random pushing already reaches everything there. A
    learner whose average end error is more than 5% above not choosing's in a plain world
    is flagged.

Luck. A learner no better than the do-nothing critic still ends lower in at least 7 of 10
runs 17% of the time in one world, and 3% of the time in both. Over 27 learners, about one
could "work" by luck alone. The verdict says so.
"""

from math import comb

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy.lib.stride_tricks import sliding_window_view

from report_map import END_ROUNDS
from report_step1 import FAINT_INK, GRID, INK, PAPER, SOFT_INK, tidy

RESULTS_FILE = "results/comparison.npz"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
GROUPS = ("our bets", "bar to beat", "baselines", "bookkeeping")
NARROW, PLAIN = ("narrow 6", "narrow 24"), ("6", "24")
LAST_ARRIVAL = 40     # the last late objects arrive in this round (world.make_world)
WINS_NEEDED = 7       # runs of 10 (the reading rules)
FLAG_ABOVE = 0.05     # a plain world's end error this far above not choosing's is flagged
CAP_SWITCH = 3        # this many learners at the cap in a narrow world switch to the objects there from the start


def main():
    saved = np.load(RESULTS_FILE)
    names, groups, worlds = list(saved["learners"]), list(saved["groups"]), list(saved["worlds"])
    narrow, plain = [worlds.index(world) for world in NARROW], [worlds.index(world) for world in PLAIN]
    not_choosing, do_nothing = names.index("not choosing"), names.index("do-nothing critic")

    # Pushes saved (see the top of this file).
    share, at_cap = pushes_saved(saved["learnable"], not_choosing, first_window=LAST_ARRIVAL)
    measured_on = "all the learnable objects"
    if max(at_cap[:, w].sum() for w in narrow) >= CAP_SWITCH:
        share, at_cap = pushes_saved(saved["from_start"], not_choosing, first_window=0)
        measured_on = "only the learnable objects on the table from the first round"

    # The reading rules.
    end = saved["learnable"][..., -END_ROUNDS:].mean(axis=-1)    # [learner, world, run]
    against = end.mean(axis=-1) / end[not_choosing].mean(axis=-1)  # [learner, world]: end error against not choosing
    beats_do_nothing = (end < end[do_nothing]).sum(axis=-1)        # [learner, world]: runs it ends lower
    in_group = {group: [l for l, its_group in enumerate(groups) if its_group == group] for group in GROUPS}
    works = [l for l in range(len(names))
             if l != do_nothing and all(beats_do_nothing[l, w] >= WINS_NEEDED for w in narrow)]
    best_bar = min(in_group["bar to beat"], key=lambda l: against[l, narrow].mean())
    best_bet = min(in_group["our bets"], key=lambda l: against[l, narrow].mean())
    beats_best_bar = (end < end[best_bar]).sum(axis=-1)
    bets_win = [l for l in in_group["our bets"] if all(beats_best_bar[l, w] >= WINS_NEEDED for w in narrow)]
    flagged = [l for l in range(len(names)) if any(against[l, w] > 1 + FLAG_ABOVE for w in plain)]

    # Not part of the rules fixed before the run: the same count against not choosing.
    beats_not_choosing = (end < end[not_choosing]).sum(axis=-1)
    in_narrow = [l for l in range(len(names)) if all(beats_not_choosing[l, w] >= WINS_NEEDED for w in narrow)]
    everywhere = [l for l in in_narrow if all(beats_not_choosing[l, w] >= WINS_NEEDED for w in plain)]

    toys = saved["toy_share"].mean(axis=(1, 2, 3))  # [learner]: over the worlds, runs and rounds
    print_table(names, in_group, worlds, share, at_cap, against, beats_do_nothing, beats_not_choosing, toys, works,
                do_nothing, measured_on)
    print_verdict(names, works, best_bar, bets_win, flagged, in_narrow, everywhere, not_choosing,
                  compared=len(names) - 1)
    draw_pushes_saved(names, in_group, worlds, share, works, do_nothing, measured_on)
    shown = [(not_choosing, INK, "-", "not choosing"), (do_nothing, FAINT_INK, "--", "do-nothing critic"),
             (names.index("raw error"), ORANGE, "-", "raw error")]
    if best_bar != names.index("raw error"):
        shown.append((best_bar, GREEN, "-", f"best of the bar to beat: {names[best_bar]}"))
    shown.append((best_bet, BLUE, "-", f"best of our bets: {names[best_bet]}"))
    draw_curves(saved["learnable"], worlds, shown)


def pushes_saved(lines, reference, first_window):
    """The share of pushes each learner saves to get down to the reference's end error.

    lines is [learner, world, run, round]. Each learner's line is averaged over the runs and
    smoothed: the window that starts in round i is the average of rounds i to i + 4. The
    first window, from first_window on, that is at most the reference's end error is where
    the learner gets there, after i + 5 rounds. Returns, as [learner, world]: the share of
    pushes saved against the reference (a gap, not-a-number, for never), and whether the
    learner got there in the very first window searched, the most this can show.
    """
    smooth = sliding_window_view(lines.mean(axis=2), END_ROUNDS, axis=-1).mean(axis=-1)  # [learner, world, window]
    target = smooth[reference, :, -1]  # the reference's end error, in each world
    low_enough = smooth[..., first_window:] <= target[None, :, None]
    window = np.where(low_enough.any(axis=-1), first_window + low_enough.argmax(axis=-1), -1)
    rounds_needed = window + END_ROUNDS
    share = np.where(window >= 0, 1 - rounds_needed / rounds_needed[reference], np.nan)
    return share, window == first_window


def print_table(names, in_group, worlds, share, at_cap, against, beats_do_nothing, beats_not_choosing, toys, works,
                do_nothing, measured_on):
    print("Pushes saved: how many fewer pushes a learner needs to get down to the error not choosing ends at,")
    print(f"measured on {measured_on}. never: it does not get there in 70 rounds. *: at the cap, the most")
    print("this can show. End error: against not choosing's, negative is better. Beats do-nothing, beats not")
    print("choosing: in how many of the 10 runs it ends lower. Toys: share of its pushes on toys (1 in 6 objects).")
    print("A learner marked + works by the reading rules.\n")
    order = [worlds.index(world) for world in (*NARROW, *PLAIN)]
    narrow = order[:len(NARROW)]
    titles = "".join(f"{short(worlds[w]):>8s}" for w in order)
    narrow_titles = "".join(f"{short(world):>8s}" for world in NARROW)
    print(f"{'':38s}{'pushes saved':^32s}  {'end error':^32s}  {'beats do-nothing':^16s}  "
          f"{'beats not choosing':^18s}  toys")
    print(f"{'':38s}{titles}  {titles}  {narrow_titles}    {narrow_titles}")
    for group, members in in_group.items():
        print(group)
        for l in members:
            saved_cells = "".join(f"{saved_text(share[l, w], at_cap[l, w]):>8s}" for w in order)
            end_cells = "".join(f"{against[l, w] - 1:+8.0%}" for w in order)
            do_nothing_cells = "".join(f"{'-' if l == do_nothing else f'{beats_do_nothing[l, w]}/10':>8s}"
                                       for w in narrow)
            not_choosing_cells = "".join(f"{f'{beats_not_choosing[l, w]}/10':>8s}" for w in narrow)
            mark = "+" if l in works else " "
            print(f"  {mark} {names[l]:34s}{saved_cells}  {end_cells}  {do_nothing_cells}    {not_choosing_cells}"
                  f"  {toys[l]:4.0%}")
    print()


def print_verdict(names, works, best_bar, bets_win, flagged, in_narrow, everywhere, not_choosing, compared):
    one_world = sum(comb(10, wins) for wins in range(WINS_NEEDED, 11)) / 2**10
    both = one_world**2
    print("Verdict, by the rules fixed before the run")
    print(f"  Works (ends below the do-nothing critic in at least {WINS_NEEDED} of 10 runs, in both narrow worlds): "
          f"{listed(names, works)}.")
    if not_choosing in works:
        print("  Not choosing passes this rule too: the do-nothing critic ends worse than not choosing. So a learner")
        print("  no better than not choosing can pass, and the luck count below is too low.")
    print(f"  The best of the bar to beat is {names[best_bar]}. Bets that beat it the same way: "
          f"{listed(names, bets_win)}.")
    print(f"  Flagged (a plain world's end error more than {FLAG_ABOVE:.0%} above not choosing's): "
          f"{listed(names, flagged)}.")
    print(f"  Luck: with no real difference, {WINS_NEEDED} or more of 10 runs happen {one_world:.0%} of the time in "
          f"one world and {both:.0%} in both, so over {compared} learners about {compared * both:.1f} would pass "
          "by luck.")
    print("Not fixed before the run: the same count against not choosing")
    print(f"  Ends below not choosing in at least {WINS_NEEDED} of 10 runs in both narrow worlds: "
          f"{listed(names, in_narrow)}.")
    print(f"  ... and in both plain worlds as well: {listed(names, everywhere)}.")


def listed(names, learners):
    return "; ".join(names[l] for l in learners) if learners else "none"


def saved_text(value, at_cap):
    if np.isnan(value):
        return "never"
    return f"{value:+.0%}" + ("*" if at_cap else "")


def short(world):
    return world.replace("narrow ", "n")


def draw_pushes_saved(names, in_group, worlds, share, works, do_nothing, measured_on):
    """One row per learner, one panel per world size. Filled dot: narrow world. Hollow dot: plain world."""
    rows, labels, headers = [], [], []
    for group, members in in_group.items():
        labels.append(group)
        headers.append(True)
        rows.append(None)
        for l in members:
            labels.append(("✓ " if l in works else "") + names[l])
            headers.append(False)
            rows.append(l)
    y = -np.arange(len(rows), dtype=float)
    learners = np.array([row is not None for row in rows])
    picked = [row for row in rows if row is not None]

    finite = share[np.isfinite(share)]
    low, high = min(-0.2, finite.min()) - 0.05, max(0.2, finite.max()) + 0.05
    never_at = low - 0.12  # "never" gets a column of its own, left of the scale
    ticks = [never_at] + [tick / 10 for tick in range(int(np.ceil(low * 10)), int(np.floor(high * 10)) + 1)]

    figure, panels = plt.subplots(1, 2, figsize=(12, 10), sharey=True, facecolor=PAPER)
    for panel, size in zip(panels, ("6", "24")):
        narrow_world, plain_world = worlds.index(f"narrow {size}"), worlds.index(size)
        for w, filled, shift in ((plain_world, False, -0.13), (narrow_world, True, 0.13)):  # apart, so both show
            values = share[picked, w]
            x = np.where(np.isfinite(values), values, never_at)
            panel.scatter(x, y[learners] + shift, s=46, zorder=3, linewidths=1.5, edgecolors=BLUE,
                          facecolors=BLUE if filled else PAPER)
        panel.axvline(0, color=INK, linewidth=1)
        if np.isfinite(share[do_nothing, narrow_world]):  # a learner that never gets there has no line
            panel.axvline(share[do_nothing, narrow_world], color=FAINT_INK, linewidth=1.5, linestyle="--")
        panel.axvline((never_at + low) / 2, color=GRID, linewidth=1)
        panel.set_xlim(never_at - 0.06, high)
        panel.set_xticks(ticks, ["never"] + [f"{tick:+.0%}" if tick else "0" for tick in ticks[1:]])
        panel.set_yticks(y, labels)
        tidy(panel)
        panel.grid(axis="x", color=GRID, linewidth=1)
        panel.set_title(f"{size} objects", loc="left", fontsize=11, color=INK)
        panel.set_xlabel("pushes saved against not choosing (right is better)", color=SOFT_INK, fontsize=9)
    for label, header in zip(panels[0].get_yticklabels(), headers):
        label.set_fontweight("bold" if header else "normal")
        label.set_color(INK if header else SOFT_INK)

    legend = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="narrow world"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=PAPER, markeredgecolor=BLUE, markeredgewidth=1.5,
               markersize=7, label="plain world"),
        Line2D([], [], color=INK, linewidth=1, label="not choosing"),
    ]
    if np.isfinite(share[do_nothing, [worlds.index(world) for world in NARROW]]).any():
        legend.append(Line2D([], [], color=FAINT_INK, linewidth=1.5, linestyle="--",
                             label="do-nothing critic, narrow world"))
    panels[0].legend(handles=legend, loc="lower left", bbox_to_anchor=(0, 1.04), ncol=4, frameon=False, fontsize=9)
    figure.text(0.01, 0.005, f"Measured on {measured_on}. ✓: works by the reading rules (beats the do-nothing "
                "critic in at least 7 of 10 runs, in both narrow worlds).", color=SOFT_INK, fontsize=8)
    figure.tight_layout(rect=(0, 0.02, 1, 1))
    figure.savefig(RESULTS_FILE.replace(".npz", ".png"), dpi=160)


def draw_curves(lines, worlds, shown):
    """The true error round by round in the two narrow worlds, averaged over the runs, for a few learners."""
    figure, panels = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=PAPER)
    rounds = np.arange(1, lines.shape[-1] + 1)
    for panel, world in zip(panels, NARROW):
        w = worlds.index(world)
        for arrival in (20, LAST_ARRIVAL):
            panel.axvline(arrival + 1, color=GRID, linewidth=1)
        for l, color, style, label in shown:
            panel.plot(rounds, lines[l, w].mean(axis=0), color=color, linestyle=style, linewidth=2, label=label)
        panel.set_yscale("log")
        panel.minorticks_off()
        low, high = panel.get_ylim()
        ticks = [tick for tick in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2) if low <= tick <= high]
        panel.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
        tidy(panel)
        panel.grid(axis="y", color=GRID, linewidth=1)
        panel.set_title(f"{world.removeprefix('narrow ')} objects, narrow", loc="left", fontsize=11, color=INK)
        panel.set_xlabel("round (grey lines: late objects arrive)", color=SOFT_INK, fontsize=9)
    panels[0].set_ylabel("true error on the learnable objects\n(log scale, lower is better)", color=SOFT_INK,
                         fontsize=9)
    figure.legend(*panels[0].get_legend_handles_labels(), loc="upper left", ncol=3, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(RESULTS_FILE.replace(".npz", "_curves.png"), dpi=160)


if __name__ == "__main__":
    main()
