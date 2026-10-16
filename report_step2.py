"""Read the results of step 2: print the scoreboard and draw one figure.

Run it with:   python3 report_step2.py      (after run_step2.py has written the results)

The scoreboard is the true error: the miss against the world's rules, on fixed test
pushes. It is shown for each object, and summed up in two ways:

    learnable   every object but the toy, counted equally
    all         the toy counted too. Late in a run, the toy's memorized error is the
                largest part of this line, so this line rewards fixing the toy.

The pass rule, agreed before the run: a learner passes if, in both training settings,
it ends with lower true error than random pushing on the learnable objects, and every
new object gets at least its random-pushing share of pushes in its first few rounds.

Random pushing gives wood 40% of the pushes. So the report also compares every learner
with even pushing, a baseline that spreads its random pushes evenly over the objects.
This comparison was added after the first run and is not part of the pass rule.
"""

import matplotlib.pyplot as plt
import numpy as np

from report_step1 import FAINT_INK, GRID, INK, PAPER, SOFT_INK, tidy

RESULTS_FILE = "results/step2.npz"
END_ROUNDS = 5        # "the end of the run" is the average over the last this-many rounds
NEWCOMER_ROUNDS = 5   # a new object is checked over its first this-many rounds on the table

# The two baselines are grey, one solid and one dashed. The others take the first colours of step 1's palette.
LEARNER_COLOURS = {
    "random pushing": FAINT_INK, "even pushing": FAINT_INK,
    "raw error": "#2a78d6", "repeat agreement": "#eb6834", "forward progress": "#1baf7a",
}
DASHED = ("even pushing",)


def main():
    saved = np.load(RESULTS_FILE)
    learners, training = list(saved["learners"]), list(saved["training"])
    objects, appears_at, share = list(saved["objects"]), saved["appears_at"], saved["share"]
    true_error = saved["true_error"]    # [learner, training setting, run, round, object]
    pushes_made = saved["pushes_made"]  # [learner, training setting, run, round, object]
    toy = objects.index("toy")

    for t, steps in enumerate(training):
        print(f"===== Training {steps} steps per round =====\n")
        print_scoreboard(true_error[:, t], learners, objects, toy)
        print_pushes(pushes_made[:, t], learners, objects, appears_at)
    print_pass_check(true_error, pushes_made, learners, training, objects, appears_at, share, toy)
    draw_curves(true_error, learners, training, appears_at, toy)


def summary_lines(true_error, toy):
    """The two summary lines, for every round: learnable objects only, and all objects.

    Objects not yet on the table are gaps, and are skipped.
    """
    learnable = np.nanmean(np.delete(true_error, toy, axis=-1), axis=-1)
    everything = np.nanmean(true_error, axis=-1)
    return learnable, everything


def print_scoreboard(true_error, learners, objects, toy):
    """true_error is [learner, run, round, object] for one training setting."""
    print("True error per object, averaged over the run (lower is better):")
    print(f"  {'':18s}" + "".join(f"{name:>9s}" for name in objects))
    for learner, error in zip(learners, true_error):
        per_object = np.nanmean(error.mean(axis=0), axis=0)  # average the runs, then the rounds on the table
        print(f"  {learner:18s}" + "".join(f"{value:9.4f}" for value in per_object))

    learnable, everything = summary_lines(true_error, toy)  # [learner, run, round]
    end = learnable[:, :, -END_ROUNDS:].mean(axis=(1, 2))   # learnable objects, at the end of the run
    end_of_random, end_of_even = end[learners.index("random pushing")], end[learners.index("even pushing")]
    print("\nSummed up:          over the run          | at the end of the run")
    print(f"  {'':18s}{'learnable':>10s}{'all':>8s}     |{'learnable':>10s}{'all':>8s}"
          f"{'vs random pushing':>20s}{'vs even pushing':>18s}")
    for l, learner in enumerate(learners):
        print(f"  {learner:18s}{learnable[l].mean():10.4f}{everything[l].mean():8.4f}     |"
              f"{end[l]:10.4f}{everything[l][:, -END_ROUNDS:].mean():8.4f}"
              f"{end[l] / end_of_random - 1:20.0%}{end[l] / end_of_even - 1:18.0%}")
    print()


def print_pushes(pushes_made, learners, objects, appears_at):
    """Where each learner's pushes went, by phase. pushes_made is [learner, run, round, object]."""
    starts = [0, *sorted(int(r) for r in appears_at if r > 0)]
    ends = [*starts[1:], pushes_made.shape[2]]
    print("Where the pushes went (share of all pushes):")
    for start, end in zip(starts, ends):
        print(f"  rounds {start} to {end - 1}")
        print(f"    {'':18s}" + "".join(f"{name:>9s}" for name in objects))
        for learner, made in zip(learners, pushes_made):
            counts = made[:, start:end].sum(axis=(0, 1))
            print(f"    {learner:18s}" + "".join(f"{share:9.0%}" for share in counts / counts.sum()))
    print()


def print_pass_check(true_error, pushes_made, learners, training, objects, appears_at, share, toy):
    print("Pass rule: in both settings, lower true error than random pushing on the learnable objects at the")
    print(f"end of the run, and every new object gets at least its random-pushing share in its first {NEWCOMER_ROUNDS} rounds.\n")
    learnable, _ = summary_lines(true_error, toy)              # [learner, training setting, run, round]
    end = learnable[..., -END_ROUNDS:].mean(axis=(-2, -1))     # [learner, training setting]
    random, even = learners.index("random pushing"), learners.index("even pushing")

    for l, learner in enumerate(learners):
        if l in (random, even):
            continue
        lower = [end[l, t] < end[random, t] for t in range(len(training))]
        lower_than_even = [end[l, t] < end[even, t] for t in range(len(training))]
        notes = []
        fair = []
        for t, steps in enumerate(training):
            for obj in np.flatnonzero(appears_at > 0):
                start = appears_at[obj]
                window = pushes_made[l, t, :, start:start + NEWCOMER_ROUNDS]  # [run, round, object]
                got = window[..., obj].sum() / window.sum()
                fair_share = share[obj] / share[appears_at <= start].sum()
                fair.append(got >= fair_share)
                notes.append(f"{objects[obj]} {got:.0%} (fair {fair_share:.0%}) at {steps} steps")
        verdict = "PASSES" if all(lower) and all(fair) else "fails"
        print(f"  {learner}: {verdict}")
        print("    lower than random pushing at the end: "
              + ", ".join(f"{steps} steps {'yes' if ok else 'no'}" for steps, ok in zip(training, lower)))
        print("    new objects' share of pushes: " + "; ".join(notes))
        print("    (not in the rule) lower than even pushing at the end: "
              + ", ".join(f"{steps} steps {'yes' if ok else 'no'}" for steps, ok in zip(training, lower_than_even)))
    print()


def draw_curves(true_error, learners, training, appears_at, toy):
    """True error over the run: one row per training setting, learnable objects left, all objects right."""
    learnable, everything = summary_lines(true_error, toy)  # [learner, training setting, run, round]
    rounds = np.arange(true_error.shape[3])

    figure, panels = plt.subplots(len(training), 2, figsize=(11, 7), sharex=True, facecolor=PAPER)
    for t, steps in enumerate(training):
        for column, (what, line) in enumerate((("learnable objects", learnable), ("all objects, toy included", everything))):
            panel = panels[t, column]
            for l, learner in enumerate(learners):
                panel.plot(rounds, line[l, t].mean(axis=0), color=LEARNER_COLOURS[learner], linewidth=2,
                           linestyle="--" if learner in DASHED else "-", label=learner)
            for start in appears_at[appears_at > 0]:
                panel.axvline(start, color=FAINT_INK, linewidth=1)
            panel.set_yscale("log")
            panel.set_title(f"{what}, training {steps} steps per round", loc="left", fontsize=9, color=SOFT_INK)
            tidy(panel)
            panel.grid(axis="y", color=GRID, linewidth=1)
    for panel in panels[-1]:
        panel.set_xlabel("round (steel arrives at 20, the spinner at 40)", color=SOFT_INK, fontsize=9)
    figure.legend(*panels[0, 0].get_legend_handles_labels(), loc="upper right", ncol=len(learners), frameon=False, fontsize=9)
    figure.suptitle("True error over the run (lower is better)", x=0.01, ha="left", fontsize=11, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig("results/step2_true_error.png", dpi=160)


if __name__ == "__main__":
    main()
