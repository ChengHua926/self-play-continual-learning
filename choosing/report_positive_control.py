"""Read the positive control (run_positive_control.py): does the measuring stick reward exploring?

Run it with:   ../.venv/bin/python report_positive_control.py     (after run_positive_control.py)

Knowledge (test_set.py) is the share of the game's movement the world model predicts, room by
room, every room counting the same: 0 is no better than guessing "nothing moves", 1 is perfect.
"The end" is the average of the last 5 measurements, the last 500,000 frames.

It prints the rule fixed before the run and its verdict, then checks that do not depend on it:
    - across all 20 jump-back runs, does knowledge go with the rooms a run reached? If the
      measuring stick rewards exploring, runs that reached more rooms should know more.
    - uniform choice against random play, which almost never leaves the first room
    - knowledge in the test rooms a run reached, and in those it did not
    - the verdict again without the two thin test rooms (under 25 places), and with bytes that
      barely move weighed by their own movement, without the floor of 1 (test_set.py)
    - every cell-counts run against every uniform run (100 pairs), since run k of the two
      learners shares only its starting weights
It draws two panels: knowledge and rooms reached, against frames played.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from report_step1 import GRID, INK, PAPER, SOFT_INK, tidy

RESULTS_FILE = "results/positive_control.npz"
END = 5           # "the end" is the average of the last 5 measurements
WINS_NEEDED = 7   # of the 10 runs
THIN_ROOM = 25    # test rooms with fewer places than this are left out in the robustness check
COLORS = {"uniform choice": INK, "cell counts": "#2a78d6", "random play": "#eb6834"}


def main():
    saved = np.load(RESULTS_FILE)
    learners = list(saved["learners"])
    uniform, counts, random_play = (learners.index(name) for name in ("uniform choice", "cell counts", "random play"))
    rooms, frames = saved["rooms"], saved["frames"]                       # [learner, run, measurement]
    no_change, places_per_room = saved["no_change"], saved["places_per_room"]  # [test room]
    room_knowledge = np.maximum(1 - saved["room_errors"] / no_change, 0)  # [learner, run, measurement, test room]
    assert np.isfinite(room_knowledge).all(), "a world model diverged: its knowledge is not a number"
    knowledge = room_knowledge.mean(axis=-1)
    end = knowledge[..., -END:].mean(axis=-1)                             # [learner, run]
    rooms_end, reached_end = rooms[..., -1], saved["rooms_reached"][:, :, -1]  # [learner, run], [learner, run, test room]

    print(f"Knowledge at the end: the share of the game's movement the world model predicts, every room counting\n"
          f"the same ({len(no_change)} test rooms). Rooms reached by the end, of all rooms; test rooms reached.\n")
    print(f"{'':16s}{'knowledge':>11s}{'rooms':>8s}{'test rooms':>12s}{'training steps':>16s}   knowledge, run by run")
    for l, learner in enumerate(learners):
        print(f"{learner:16s}{end[l].mean():11.1%}{rooms_end[l].mean():8.1f}{reached_end[l].sum(axis=-1).mean():12.1f}"
              f"{saved['training_steps'][l, :, -1].mean():16,.0f}   " + " ".join(f"{e:.0%}" for e in end[l]))

    wins = int((end[counts] > end[uniform]).sum())
    more_rooms = rooms_end[counts].mean() > rooms_end[uniform].mean()
    passed = wins >= WINS_NEEDED and more_rooms
    print(f"\nThe rule: cell counts ends with more knowledge than uniform choice in at least {WINS_NEEDED} of 10 runs "
          f"(it does in {wins}),\nand reaches more rooms on average ({rooms_end[counts].mean():.1f} against "
          f"{rooms_end[uniform].mean():.1f}: {'yes' if more_rooms else 'no'}).")
    print(f"Verdict: the positive control {'PASSES' if passed else 'FAILS'}.\n")

    print("Checks that do not depend on the rule:")
    jump_back = [uniform, counts]
    together = end[jump_back].ravel()
    rho_rooms = spearmanr(together, rooms_end[jump_back].ravel()).statistic
    rho_test_rooms = spearmanr(together, reached_end[jump_back].sum(axis=-1).ravel()).statistic
    print(f"  Over the 20 jump-back runs, knowledge goes with rooms reached: rank correlation {rho_rooms:+.2f} "
          f"(with test rooms reached: {rho_test_rooms:+.2f}).")
    print(f"  Uniform choice against random play: more knowledge in {int((end[uniform] > end[random_play]).sum())} of 10 "
          f"runs ({end[uniform].mean():.1%} against {end[random_play].mean():.1%}).")
    final = room_knowledge[..., -END:, :].mean(axis=-2)  # [learner, run, test room]
    for l in jump_back + [random_play]:
        inside, outside = final[l][reached_end[l]], final[l][~reached_end[l]]
        print(f"  {learners[l]}: knowledge in test rooms it reached {inside.mean():.1%} ({len(inside)} room-runs), "
              f"in those it did not {outside.mean() if len(outside) else float('nan'):.1%} ({len(outside)}).")
    thick = places_per_room >= THIN_ROOM
    end_thick = room_knowledge[..., -END:, :][..., thick].mean(axis=(-2, -1))
    print(f"  Without the {int((~thick).sum())} thin test rooms: cell counts ends above uniform choice in "
          f"{int((end_thick[counts] > end_thick[uniform]).sum())} of 10 runs.")
    movement = saved["no_change_misses"].mean(axis=0)  # each byte's movement, without the floor of 1
    unfloored = knowledge_from(saved["room_squared_misses"] / movement, saved["no_change_misses"] / movement)
    end_unfloored = unfloored[..., -END:].mean(axis=-1)
    print(f"  Without the floor of 1 on movement: cell counts ends above uniform choice in "
          f"{int((end_unfloored[counts] > end_unfloored[uniform]).sum())} of 10 runs "
          f"(knowledge {end_unfloored[counts].mean():.1%} against {end_unfloored[uniform].mean():.1%}).")
    pairs = (end[counts][:, None] > end[uniform][None, :]).mean()
    print(f"  Every cell-counts run against every uniform run: cell counts knows more in {pairs:.0%} of 100 pairs.")
    at_2m = np.argmin(np.abs(frames[0, 0] - 2_000_000))
    print("  At about 2 million frames, knowledge: " + ", ".join(
        f"{learner} {knowledge[l, :, at_2m].mean():.1%}" for l, learner in enumerate(learners)) + ".")
    draw(learners, knowledge, rooms, frames)


def knowledge_from(weighted_misses, weighted_no_change):
    """Knowledge from squared misses already divided by each byte's movement: [..., room, byte] and [room, byte]."""
    room_errors, no_change = weighted_misses.mean(axis=-1), weighted_no_change.mean(axis=-1)
    return np.maximum(1 - room_errors / no_change, 0).mean(axis=-1)


def draw(learners, knowledge, rooms, frames):
    figure, panels = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=PAPER)
    millions = frames[0, 0] / 1e6
    for panel, line, label in ((panels[0], knowledge, "knowledge of the game (higher is better)"),
                               (panels[1], rooms, "rooms reached")):
        for l, learner in enumerate(learners):
            for run in line[l]:
                panel.plot(millions, run, color=COLORS[learner], linewidth=0.6, alpha=0.3)
            panel.plot(millions, line[l].mean(axis=0), color=COLORS[learner], linewidth=2.2, label=learner)
        tidy(panel)
        panel.grid(axis="y", color=GRID, linewidth=1)
        panel.set_xlabel("frames played (millions)", color=SOFT_INK, fontsize=9)
        panel.set_ylabel(label, color=SOFT_INK, fontsize=9)
    panels[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    figure.legend(*panels[0].get_legend_handles_labels(), loc="upper left", ncol=3, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(RESULTS_FILE.replace(".npz", ".png"), dpi=160)


if __name__ == "__main__":
    main()
