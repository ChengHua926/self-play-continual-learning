"""Read the checks of the setup (run_checks.py), with rules fixed before the run.

Run it with:   ../.venv/bin/python report_checks.py      (after run_checks.py)

Knowledge (test_set.py) is the share of the game's movement the world model predicts, room by
room, every room counting the same. Each run is read by its knowledge averaged over the whole
run, every 100,000 frames from the start to 8 million: a run that learns sooner knows more along
the way. The end (the last 500,000 frames) is printed beside it.

Checks with answers we know. Each must pass, or the setup has a mistake to find first:
    exploring matters   uniform choice knows more than random play in at least 7 of 10 runs
    choosing matters    uniform choice knows more than most visited in at least 7 of 10 runs
(The third, that the world model can learn every room, is check_world_model.py.)

Is there room to win? Not a check with a known answer:
    room oracle         knows more than uniform choice in at least 7 of 10 runs
If even this rule, which uses the room numbers, cannot beat uniform choice, a fair learner is
unlikely to, and the full comparison is probably not worth running at this budget. (The oracle
is one rule of thumb, under the same candidates as every rule, so it does not prove the limit.)

Reported as a result, not a check: cell counts against uniform choice.
The report also shows, for each rule, rooms reached, when it left the first room, places saved,
and the training it got (the same for all, one step per 50 steps played).
"""

import matplotlib.pyplot as plt
import numpy as np

from report_step1 import GRID, INK, PAPER, SOFT_INK, tidy

RESULTS_FILE = "results/checks.npz"
END = 5
WINS_NEEDED = 7
COLORS = {"uniform choice": INK, "random play": "#eb6834", "most visited": "#898781", "room oracle": "#1baf7a",
          "cell counts": "#2a78d6"}


def main():
    saved = np.load(RESULTS_FILE)
    rules = list(saved["rules"])
    knowledge = np.maximum(1 - saved["room_errors"] / saved["no_change"], 0).mean(axis=-1)  # [rule, run, measurement]
    assert np.isfinite(knowledge).all(), "a world model diverged: its knowledge is not a number"
    whole, end = knowledge.mean(axis=-1), knowledge[..., -END:].mean(axis=-1)                # [rule, run]
    rooms, frames = saved["rooms"], saved["frames"]
    uniform = rules.index("uniform choice")

    print("Knowledge averaged over the whole run, and at the end; rooms reached by the end; when the first room\n"
          "was left (median over the runs that left it); places saved; training steps.\n")
    print(f"{'':16s}{'whole run':>10s}{'end':>8s}{'rooms':>7s}{'left room 1':>13s}{'places':>8s}{'training':>10s}")
    for r, rule in enumerate(rules):
        left = [frames[r, run, np.flatnonzero(rooms[r, run] > 1)[0]] / 1e6
                for run in range(rooms.shape[1]) if (rooms[r, run] > 1).any()]
        left_text = f"{np.median(left):.1f}M ({len(left)})" if left else "never"
        print(f"{rule:16s}{whole[r].mean():10.1%}{end[r].mean():8.1%}{rooms[r, :, -1].mean():7.1f}{left_text:>13s}"
              f"{saved['places'][r, :, -1].mean():8.0f}{saved['training_steps'][r, :, -1].mean():10,.0f}")

    def wins(a, b):
        return int((whole[rules.index(a)] > whole[rules.index(b)]).sum())

    print("\nChecks with answers we know (whole-run knowledge, runs of 10):")
    exploring = wins("uniform choice", "random play")
    choosing = wins("uniform choice", "most visited")
    print(f"  exploring matters: uniform choice knows more than random play in {exploring} of 10 runs: "
          f"{'PASSES' if exploring >= WINS_NEEDED else 'FAILS'}")
    print(f"  choosing matters: uniform choice knows more than most visited in {choosing} of 10 runs: "
          f"{'PASSES' if choosing >= WINS_NEEDED else 'FAILS'}")
    oracle = wins("room oracle", "uniform choice")
    print(f"\nIs there room to win? The room oracle knows more than uniform choice in {oracle} of 10 runs: "
          f"{'yes' if oracle >= WINS_NEEDED else 'no'}.")
    counts = wins("cell counts", "uniform choice")
    print(f"Result: cell counts knows more than uniform choice in {counts} of 10 runs "
          f"({whole[rules.index('cell counts')].mean():.1%} against {whole[uniform].mean():.1%} over the whole run; "
          f"at the end {end[rules.index('cell counts')].mean():.1%} against {end[uniform].mean():.1%}).")
    draw(rules, knowledge, rooms, frames)


def draw(rules, knowledge, rooms, frames):
    figure, panels = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=PAPER)
    millions = frames[0, 0] / 1e6
    for panel, line, label in ((panels[0], knowledge, "knowledge of the game (higher is better)"),
                               (panels[1], rooms.astype(float), "rooms reached")):
        for r, rule in enumerate(rules):
            for run in line[r]:
                panel.plot(millions, run, color=COLORS[rule], linewidth=0.5, alpha=0.25)
            panel.plot(millions, line[r].mean(axis=0), color=COLORS[rule], linewidth=2.2, label=rule)
        tidy(panel)
        panel.grid(axis="y", color=GRID, linewidth=1)
        panel.set_xlabel("frames played (millions)", color=SOFT_INK, fontsize=9)
        panel.set_ylabel(label, color=SOFT_INK, fontsize=9)
    panels[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    figure.legend(*panels[0].get_legend_handles_labels(), loc="upper left", ncol=5, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(RESULTS_FILE.replace(".npz", ".png"), dpi=160)


if __name__ == "__main__":
    main()
