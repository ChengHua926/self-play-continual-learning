"""Read the results of step 1: print the scores and draw the three figures.

Run it with:   python3 report_step1.py      (after run_step1.py has written the results)

Each signal is meant to become the label a critic learns from. A critic learns the
label's average for each kind of push, and the learner then goes where that average is
highest. So the question for a signal is: does its average order the objects the way
the answer key does, at least where the difference matters?

The score. In every round, take every pair of objects. The answer key says which of the
two is better to push, and by how much: the gap. The signal gets the pair right if it
orders the two the same way. The score is the sum of the gaps it gets right, divided by
the sum of all gaps. 100% is perfect, 50% is a coin flip, 0% is always backwards.

It is worked out twice:
    on average   signals and answer key are first averaged over the runs. This is what a
                 critic would end up learning, so this score decides.
    one run      each run is scored on its own, then the scores are averaged. This adds the
                 noise of a single run, which a critic has to average away.

And against two answer keys: one that looks 1 round ahead, and one that looks 10 rounds
ahead. They can disagree, because extra pushes can help soon and stop helping later.

The row "answer key vs itself" scores the answer key's first repeat against its second.
Both repeats start from the same network, memory and test pushes, so the row shows only
how much the answer key's own random draws move it. It says the answer key repeats
itself, not that it is right. A signal cannot be expected to do better than this row.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

RESULTS_FILE = "results/step1.npz"
CEILING = "answer key vs itself"
FLOOR = "raw error"

# One colour per object, in a fixed order that stays readable for colour-blind readers.
OBJECT_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
BAR_COLOUR = "#2a78d6"
PAPER, INK, SOFT_INK, FAINT_INK, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def main():
    saved = np.load(RESULTS_FILE)
    objects, signal_names = list(saved["objects"]), list(saved["signal_names"])
    appears_at = saved["appears_at"]

    # Keep only the rounds that have an answer key. The first few do not: there,
    # "progress over time" has no past self to compare with yet.
    judged = ~np.isnan(saved["answer_key"]).all(axis=(0, 2, 3, 4))
    rounds = np.flatnonzero(judged)
    signals = saved["signals"][:, judged]     # [run, round, object, signal]
    gains = saved["answer_key"][:, judged]    # [run, round, object, repeat, rounds ahead]
    horizons = (1, gains.shape[-1])           # the answer key looking 1 round ahead, and the most rounds ahead

    phases = make_phases(rounds, objects, appears_at)
    print_gain_by_horizon(gains, objects, phases)

    scores = {}
    for horizon in horizons:
        key_repeats = gains[..., horizon - 1]  # [run, round, object, repeat]
        scores[horizon] = score_everything(signals, signal_names, key_repeats.mean(axis=3), key_repeats, phases)
        print_scores(scores[horizon], phases, horizon)

    draw_gain_by_horizon(gains, objects, phases)
    draw_scores(scores, phases)
    answer_keys = {horizon: average(gains[..., horizon - 1].mean(axis=3)) for horizon in horizons}
    draw_curves(average(signals), answer_keys, rounds, objects, signal_names, appears_at)


def average(values, axis=0):
    """An average that skips the gaps (numbers that were not measured)."""
    return np.ma.masked_invalid(values).mean(axis=axis).filled(np.nan)


def make_phases(rounds, objects, appears_at):
    """A phase starts each time a new object is put on the table. The last phase is the whole run.

    A phase is a name and a yes/no per round: is this round part of it?
    """
    starts = [int(rounds[0]), *sorted(int(r) for r in appears_at if r > 0)]
    ends = [*starts[1:], int(rounds[-1]) + 1]
    phases = []
    for start, end in zip(starts, ends):
        newcomers = [objects[o] for o in np.flatnonzero(appears_at == start)]
        name = f"rounds {start} to {end - 1}" + (f": {' and '.join(newcomers)} is new" if newcomers else "")
        phases.append((name, (rounds >= start) & (rounds < end)))
    phases.append(("all rounds", np.ones(len(rounds), dtype=bool)))
    return phases


def gain_by_horizon(gains, in_phase):
    """[object, rounds ahead]: the answer key averaged over runs, repeats and the rounds of a phase."""
    per_round = gains[:, in_phase].mean(axis=3)              # [run, round, object, rounds ahead]
    return average(per_round.reshape(-1, *per_round.shape[2:]))


def pair_score(signal, answer_key):
    """The share of what is at stake that a signal orders right. Both arguments are [round, object].

    Within a round, every pair of objects is one question: which of the two is better to push?
    What is at stake in that question is the answer key's gap between the two.
    """
    right = at_stake = 0.0
    for signal_now, key_now in zip(signal, answer_key):
        measured = np.flatnonzero(~np.isnan(signal_now) & ~np.isnan(key_now))
        for first, second in itertools.combinations(measured, 2):
            gap = abs(key_now[first] - key_now[second])
            at_stake += gap
            if np.sign(signal_now[first] - signal_now[second]) == np.sign(key_now[first] - key_now[second]):
                right += gap
    return right / at_stake


def score_everything(signals, signal_names, answer_key, key_repeats, phases):
    """Every row of the report: {name: {"on average": [one score per phase], "one run": [...]}}.

    Arrays are indexed in two steps on purpose. In one step, numpy would move the round
    axis to the front when a yes/no mask and a single number are mixed.
    """
    runs = range(len(answer_key))
    first_repeat, second_repeat = key_repeats[..., 0], key_repeats[..., 1]
    scores = {CEILING: {
        "on average": [
            both_ways(average(first_repeat[:, in_phase]), average(second_repeat[:, in_phase]))
            for _, in_phase in phases
        ],
        "one run": [
            np.mean([both_ways(first_repeat[run, in_phase], second_repeat[run, in_phase]) for run in runs])
            for _, in_phase in phases
        ],
    }}
    for s, name in enumerate(signal_names):
        signal = signals[..., s]  # [run, round, object]
        scores[name] = {
            "on average": [
                pair_score(average(signal[:, in_phase]), average(answer_key[:, in_phase]))
                for _, in_phase in phases
            ],
            "one run": [
                np.mean([pair_score(signal[run, in_phase], answer_key[run, in_phase]) for run in runs])
                for _, in_phase in phases
            ],
        }
    return scores


def both_ways(first, second):
    """The pair score of two noisy versions of the answer key, each used once as the key."""
    return (pair_score(first, second) + pair_score(second, first)) / 2


def print_gain_by_horizon(gains, objects, phases):
    print("How much lower is the true error with extra pushes of each object? In thousandths,")
    print("averaged over the rounds of each phase. Columns: how many rounds ahead the answer key looks.\n")
    horizons = range(1, gains.shape[-1] + 1)
    for name, in_phase in phases[:-1]:
        print(name)
        print(f"  {'':10s}" + "".join(f"{h:7d}" for h in horizons))
        for obj, gain in zip(objects, gain_by_horizon(gains, in_phase)):
            if not np.isnan(gain).all():
                print(f"  {obj:10s}" + "".join(f"{1000 * g:7.2f}" for g in gain))
        print()


def print_scores(scores, phases, horizon):
    print(f"Scored against the answer key that looks {horizon} round{'s' if horizon > 1 else ''} ahead.")
    print("Does each signal order the objects like the answer key, weighted by what is at stake?")
    print("100% is perfect, 50% is a coin flip.\n")
    header = f"  {'':26s}" + "".join(f"{name.split(':')[0]:>17s}" for name, _ in phases)
    for kind in ("on average", "one run"):
        print(kind + (" (this one decides)" if kind == "on average" else ""))
        print(header)
        for name, by_kind in scores.items():
            label = name + (" (floor)" if name == FLOOR else "")
            print(f"  {label:26s}" + "".join(f"{score:17.0%}" for score in by_kind[kind]))
        print()


def draw_gain_by_horizon(gains, objects, phases):
    """One panel per phase: the real gain from extra pushes of each object, against rounds looked ahead."""
    horizons = np.arange(1, gains.shape[-1] + 1)
    figure, panels = plt.subplots(1, len(phases) - 1, figsize=(12, 3.6), sharey=True, facecolor=PAPER)
    for panel, (name, in_phase) in zip(panels, phases[:-1]):
        for obj, gain in enumerate(gain_by_horizon(gains, in_phase)):
            if not np.isnan(gain).all():
                panel.plot(horizons, gain, color=OBJECT_COLOURS[obj], linewidth=2, marker="o", markersize=4,
                           label=objects[obj])
        panel.axhline(0, color=FAINT_INK, linewidth=1)
        panel.set_title(name, loc="left", fontsize=9, color=SOFT_INK)
        panel.set_xlabel("rounds the answer key looks ahead", color=SOFT_INK, fontsize=9)
        panel.set_xticks(horizons)
        tidy(panel)
        panel.grid(axis="y", color=GRID, linewidth=1)
    handles, labels = panels[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=len(objects), frameon=False, fontsize=9)
    figure.suptitle("Real gain from extra pushes, by how far ahead you look", x=0.01, ha="left", fontsize=11, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig("results/step1_gain_by_horizon.png", dpi=160)


def draw_scores(scores, phases):
    """One row per answer key, one panel per phase, one bar per signal: the score on average."""
    horizons = list(scores)
    names = list(scores[horizons[0]])[::-1]  # reversed, so that the first row is the top bar
    colours = [FAINT_INK if name in (CEILING, FLOOR) else BAR_COLOUR for name in names]

    figure, panels = plt.subplots(len(horizons), len(phases), figsize=(14, 6.4), sharey=True, facecolor=PAPER)
    for row, horizon in enumerate(horizons):
        for column, (phase_name, _) in enumerate(phases):
            panel = panels[row, column]
            values = [scores[horizon][name]["on average"][column] for name in names]
            bars = panel.barh(names, values, height=0.5, color=colours)
            panel.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=4, color=INK, fontsize=9)
            panel.axvline(0.5, color=FAINT_INK, linewidth=1, linestyle="--")
            panel.set_title(phase_name, loc="left", fontsize=9, color=SOFT_INK)
            panel.set_xlim(0, 1.15)
            panel.xaxis.set_major_formatter(PercentFormatter(1))
            tidy(panel)
            panel.grid(axis="x", color=GRID, linewidth=1)
        looks_ahead = f"answer key looks {horizon} round{'s' if horizon > 1 else ''} ahead"
        panels[row, 0].set_ylabel(looks_ahead, color=INK, fontsize=10, labelpad=12)
    figure.suptitle(
        "Does the signal order the objects like the answer key? Score on average "
        "(dashed: coin flip; grey: the ceiling and the floor)",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    figure.tight_layout()
    figure.savefig("results/step1_scores.png", dpi=160)


def draw_curves(signals, answer_keys, rounds, objects, signal_names, appears_at):
    """Round by round: the answer keys on top, then each signal. One line per object."""
    rows = [(f"answer key, {h} round{'s' if h > 1 else ''} ahead: real gain from extra pushes", key)
            for h, key in answer_keys.items()]
    rows += [(name, signals[:, :, s]) for s, name in enumerate(signal_names)]

    figure, panels = plt.subplots(len(rows), 1, figsize=(9, 13.5), sharex=True, facecolor=PAPER)
    for panel, (title, values) in zip(panels, rows):
        for obj, name in enumerate(objects):
            measured = ~np.isnan(values[:, obj])
            panel.plot(rounds[measured], values[measured, obj], color=OBJECT_COLOURS[obj], linewidth=2, label=name)
        for start in appears_at[appears_at > 0]:
            panel.axvline(start, color=FAINT_INK, linewidth=1)
        panel.axhline(0, color=FAINT_INK, linewidth=1)
        panel.set_title(title, loc="left", fontsize=9, color=SOFT_INK)
        tidy(panel)
        panel.grid(axis="y", color=GRID, linewidth=1)

    for obj in np.flatnonzero(appears_at > 0):
        top = panels[0].get_ylim()[1]
        panels[0].text(appears_at[obj] + 0.6, top, f"{objects[obj]} appears", va="top", fontsize=8, color=SOFT_INK)
    panels[-1].set_xlabel("round", color=SOFT_INK)
    figure.legend(*panels[0].get_legend_handles_labels(), loc="upper right", ncol=len(objects), frameon=False, fontsize=9)
    figure.suptitle("Step 1, round by round", x=0.01, ha="left", fontsize=11, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig("results/step1_curves.png", dpi=160)


def tidy(panel):
    """Quiet axes: no box, soft tick labels."""
    panel.set_facecolor(PAPER)
    for side in ("top", "right", "left", "bottom"):
        panel.spines[side].set_visible(False)
    panel.tick_params(colors=SOFT_INK, labelsize=9, length=0)
    panel.set_axisbelow(True)


if __name__ == "__main__":
    main()
