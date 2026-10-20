"""Read step B: check every signal against facts built into the worlds.

Run it with:   python3 report_step_b.py      (after run_step_b.py has written the results)

Each check looks at how a signal ranks the 64 pushes of half B within one round. The
"top picks" of a round are the quarter of its pushes (16) that the signal scores highest.
A signal that knows nothing puts a quarter of any group of pushes among its top picks.
Every check is done on each run on its own; the report gives the average over the 10 runs
and its standard error.

The rules were written down before the run, and are not changed after it. They are decided
with light training (20 steps per round). The numbers for heavy training (100 steps per
round), where the toys' noise gets memorized, are printed beside them as a stress test.

1. Ignores noise. All four worlds, rounds 10 to 69.
   The share of toy pushes that are top picks.
   Pass: at most 30% in every world. (A quarter is chance.)
   The controls pass this check too, by caring about nothing. So passing it counts only
   together with the other two checks.
   Printed beside it, not part of the rule: in what share of rounds the toys' average score
   is above the other objects' average score. A critic that learns averages sees this one.

2. Notices new things. Plain worlds. Late objects arrive in rounds 20 and 40.
   After an arrival, does a newly arrived object that is not a toy beat every old object
   (a higher average score over its pushes in B), in the round it arrives or the round after?
   New toys take part on neither side.
   Pass: yes after at least 80% of the arrivals, in both plain worlds.
   Why only the plain worlds: there the new object is the one with the most left to learn.
   In a narrow world an old object's rare part can hold more, so a signal that tracks what
   is left to learn rightly fails this there (checked with the true error before the run:
   1.00 in the plain worlds, 0.65 in narrow worlds of 24 objects). Narrow worlds are shown
   for reference.
   Printed beside it, not part of the rule: the share of the new objects' pushes that are
   top picks, in those two rounds.

3. Finds rare useful pushes. Narrow worlds, rounds 5 to 69.
   For boxes, rollers and spinners in turn: the chance that a push in the object's narrow
   part ranks above a push elsewhere on the same kind of object. 0.5 means no preference.
   Pass: for every kind, in both narrow worlds, at least 0.6, and at least 0.05 above the
   same number in the plain world of the same size, which has the same objects and runs.
   Why the plain worlds: there, the "narrow parts" are just hard pushes, pushes along a
   roller's facing, and hits near a spinner's edge. Those give the biggest results, so a
   signal that simply likes big results prefers them in plain worlds too. A signal that
   never looks at results sits near 0.5 in both, by construction.

A signal passes the screen if it passes all three checks.

Printed with the checks, not part of any rule: how forecastable each signal is. A
forecaster that knows only the object (critic.object_average) forecasts each round's
scores from the object's average score in the three rounds before. 1 = the same order as
the scores, 0 = no relation. A label that cannot be forecast cannot guide a critic, as step
A found for most labels.

4. Separates noise from rare structure. For the two noise models, not the signals. Narrow
   worlds, rounds 5 to 69. A noise model guesses how scattered each push's result is.
   (a) Toys rated noisier than other objects: the chance that a toy push outranks a push
       on another object. Pass: at least 0.9.
   (b) Rare behaviour not mistaken for noise: for boxes, rollers and spinners in turn, the
       chance that a narrow-part push is rated noisier than a push elsewhere on the same
       kind of object. Pass: at most 0.6 for every kind.
   Pass: both, in both narrow worlds, with light training. The noise model trained on
   repeats learns a noise of exactly 0 for every push that is not on a toy, so it should
   pass (b) almost by construction; the question is whether the other model does.

5. The noise models as guards (no pass rule). Even pushing, 50 training steps per round,
   with no guard, with the network that predicts its noise, and with the noise model
   trained on repeats (run_step2.py). What was expected before the run: if repeats tell
   noise from rare structure, the noise model trained on repeats keeps most of the gain of
   the noise-predicting network in plain worlds, and loses its harm in narrow worlds.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.stats import mannwhitneyu, rankdata

from critic import forecast_agreement, object_average
from report_map import END_ROUNDS
from report_step1 import FAINT_INK, GRID, INK, PAPER, tidy

RESULTS_FILE = "results/step_b.npz"
GUARDS_FILE = "results/step_b_guards.npz"
LIGHT, HEAVY = 20, 100          # training steps per round: the rules are decided at LIGHT
TOP_SHARE = 0.25                # the top picks are this share of a round's pushes
NOISE_ROUNDS = slice(10, 70)    # check 1 looks at these rounds
RARE_ROUNDS = slice(5, 70)      # check 3, check 4 and the forecasts look at these rounds
ARRIVALS = (20, 40)             # the rounds in which late objects arrive (world.make_world)
TOYS_AT_MOST = 0.30             # check 1 passes if at most this share of toy pushes are top picks
NOTICED_AT_LEAST = 0.80         # check 2 passes if a new object wins after at least this share of arrivals
RARE_AT_LEAST = 0.6             # check 3 passes if the narrow parts outrank the rest at least this often,
RARE_ABOVE_PLAIN = 0.05         # and by at least this much more than in the plain world
CHECKED_KINDS = ("box", "roller", "spinner")
TOYS_NOISIER_AT_LEAST = 0.9     # check 4a
RARE_NOISIER_AT_MOST = 0.6      # check 4b

# Plain worlds are blue and narrow ones orange; the bigger world is the darker one.
WORLD_COLOURS = {"6": "#9cc3ee", "24": "#2a78d6", "narrow 6": "#f5b48f", "narrow 24": "#eb6834"}
KIND_COLOURS = ("#1baf7a", "#e87ba4", "#eda100")  # boxes, rollers, spinners: kept apart from the world colours
PASS_SHADE = "#e8f4ec"


def main():
    saved = np.load(RESULTS_FILE)
    signals, worlds = list(saved["signals"]), list(saved["worlds"])
    training = list(saved["training"])
    rng = np.random.default_rng(0)  # breaks ties among equal scores

    # results[steps][signal][check] = [(average over runs, standard error) for each world]
    results = {steps: {signal: checks(saved, saved[signal][t], t, rng) for signal in signals}
               for t, steps in enumerate(training)}
    passes = {signal: passed(results[LIGHT][signal], worlds) for signal in signals}
    print_checks(signals, worlds, results, passes)

    noise_models = list(saved["noise_models"])
    separation = {steps: {model: noise_model_checks(saved, saved[model][t], t) for model in noise_models}
                  for t, steps in enumerate(training)}
    print_noise_models(noise_models, worlds, separation)
    print_guards(np.load(GUARDS_FILE))
    draw(signals, worlds, results[LIGHT])


def checks(saved, values, t, rng):
    """Every check of one signal at one training setting: {check: [(average over runs, standard error) per world]}.

    values is [world, run, round, push]; t is the training setting.
    """
    kinds = list(saved["kinds"])
    found = {}
    for w in range(len(saved["worlds"])):
        kind, new, objects, narrow_part = (saved[name][t, w] for name in ("kind", "new", "objects", "narrow_part"))
        toy = kind == kinds.index("toy")
        new_learnable = new & ~toy
        line = {
            "toys among top picks": over_runs(top_pick_share, (values[w], toy), rounds=NOISE_ROUNDS, rng=rng),
            "toys above average": over_runs(above_average_rounds, (values[w], toy), rounds=NOISE_ROUNDS),
            "new object wins": over_runs(new_object_wins, (values[w], objects, new_learnable, new)),
            "new among top picks": over_runs(top_pick_share, (values[w], new_learnable), rounds=slice(None), rng=rng),
            "forecastable": over_runs(object_average_agreement, (values[w], objects)),
        }
        for kind_name in CHECKED_KINDS:
            of_kind = kind == kinds.index(kind_name)
            line[f"narrow {kind_name}"] = over_runs(narrow_outranks, (values[w], of_kind, narrow_part))
        for check, number in line.items():
            found.setdefault(check, []).append(number)
    return found


def over_runs(check, arrays, **options):
    """A check done on every run on its own. Returns the average over the runs, and its standard error."""
    per_run = np.array([check(*(array[run] for array in arrays), **options) for run in range(len(arrays[0]))])
    per_run = per_run[~np.isnan(per_run)]
    return per_run.mean(), per_run.std(ddof=1) / np.sqrt(len(per_run))


def top_picks(values, rng):
    """yes/no per push: is it among the top picks of its round? Ties are broken at random.

    values is [..., push]: every row is one round.
    """
    count = int(TOP_SHARE * values.shape[-1])
    highest_first = np.lexsort((rng.random(values.shape), -values), axis=-1)
    top = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(top, highest_first[..., :count], True, axis=-1)
    return top


def top_pick_share(values, group, rounds, rng):
    """The share of the pushes in group that are top picks in their round.

    values and group are [..., round, push]. Only the given rounds count, and of those only
    the rounds in which the signal was read.
    """
    values, group = values[..., rounds, :], group[..., rounds, :]
    read = ~np.isnan(values).any(axis=-1)
    top = top_picks(np.where(np.isnan(values), -np.inf, values), rng)
    return (group & top)[read].sum() / group[read].sum()


def above_average_rounds(values, group, rounds):
    """In what share of rounds the pushes in group score above the other pushes, on average."""
    values, group = values[..., rounds, :], group[..., rounds, :]
    counted = ~np.isnan(values).any(axis=-1) & group.any(axis=-1) & (~group).any(axis=-1)
    inside = np.where(group, values, 0).sum(axis=-1) / np.maximum(group.sum(axis=-1), 1)
    outside = np.where(~group, values, 0).sum(axis=-1) / np.maximum((~group).sum(axis=-1), 1)
    return (inside > outside)[counted].mean()


def new_object_wins(values, objects, new_learnable, new):
    """The share of arrivals after which a newly arrived object that is not a toy beats every
    old object, in the round it arrives or the round after.

    "Beats" means a higher average score over its pushes in B. All arrays are [round, push]
    for one run. new marks the pushes on every newly arrived object, toys included: new toys
    take part on neither side.
    """
    noticed = []
    for arrival in ARRIVALS:
        if not new_learnable[arrival].any():
            continue  # nothing learnable arrived in this round of this run
        won = False
        for r in (arrival, arrival + 1):
            if not new_learnable[r].any() or np.isnan(values[r]).any():
                continue
            average = {obj: values[r][objects[r] == obj].mean() for obj in np.unique(objects[r])}
            best_new = max(average[obj] for obj in np.unique(objects[r][new_learnable[r]]))
            best_old = max(average[obj] for obj in np.unique(objects[r][~new[r]]))
            won |= best_new > best_old
        noticed.append(won)
    return np.mean(noticed) if noticed else np.nan


def object_average_agreement(values, objects):
    """How well a forecaster that knows only the object forecasts the order of each round's scores.

    Its forecast for a push is its object's average score in the three rounds before
    (critic.object_average). values and objects are [round, push] for one run.
    """
    agreements = []
    for r in range(RARE_ROUNDS.start, RARE_ROUNDS.stop):
        past = slice(r - 3, r)
        if np.isnan(values[past]).any() or np.isnan(values[r]).any():
            continue
        forecasts = object_average(objects[r], objects[past].ravel(), values[past].ravel())
        agreements.append(forecast_agreement(forecasts, values[r]))
    return np.nanmean(agreements)


def narrow_outranks(values, of_kind, narrow_part):
    """The chance that a push in the narrow part ranks above a push elsewhere, on objects of one kind."""
    return outrank_chance(values, of_kind & narrow_part, of_kind & ~narrow_part, RARE_ROUNDS)


def outrank_chance(values, group, other, rounds):
    """The chance that a push in group ranks above a push in other. 0.5 means no preference.

    Ranks are taken within each round, among all of that round's pushes, so that rounds
    with bigger or smaller scores count alike. All arrays are [..., round, push].
    """
    values, group, other = values[..., rounds, :], group[..., rounds, :], other[..., rounds, :]
    read = ~np.isnan(values).any(axis=-1, keepdims=True)
    ranks = rankdata(values, axis=-1)
    inside, outside = ranks[group & read], ranks[other & read]
    return mannwhitneyu(inside, outside).statistic / (len(inside) * len(outside))


def noise_model_checks(saved, values, t):
    """Check 4 for one noise model at one training setting t.

    Returns, for each world, (toys rated noisier, [rare behaviour rated noisier, for each
    checked kind]), each number as (average over runs, standard error).
    """
    kinds, found = list(saved["kinds"]), []
    for w in range(len(saved["worlds"])):
        kind, narrow_part = saved["kind"][t, w], saved["narrow_part"][t, w]
        toy = kind == kinds.index("toy")
        toys_noisier = over_runs(outrank_chance, (values[w], toy, ~toy), rounds=RARE_ROUNDS)
        rare_noisier = [over_runs(narrow_outranks, (values[w], kind == kinds.index(kind_name), narrow_part))
                        for kind_name in CHECKED_KINDS]
        found.append((toys_noisier, rare_noisier))
    return found


def passed(found, worlds):
    """The three checks for one signal, with light training: (ignores noise, notices new things, finds rare useful pushes)."""
    ignores_noise = all(share <= TOYS_AT_MOST for share, _ in found["toys among top picks"])
    notices_new = all(found["new object wins"][worlds.index(size)][0] >= NOTICED_AT_LEAST for size in ("6", "24"))
    finds_rare = True
    for kind in CHECKED_KINDS:
        for size in ("6", "24"):
            narrow, _ = found[f"narrow {kind}"][worlds.index(f"narrow {size}")]
            plain, _ = found[f"narrow {kind}"][worlds.index(size)]
            finds_rare &= narrow >= RARE_AT_LEAST and narrow >= plain + RARE_ABOVE_PLAIN
    return ignores_noise, notices_new, finds_rare


def print_checks(signals, worlds, results, passes):
    mark = {True: "pass", False: "FAIL"}
    light, heavy = results[LIGHT], results[HEAVY]
    columns = "".join(f"{world:>20s}" for world in worlds)

    def table(check, as_share=True):
        for signal in signals:
            cells = ""
            for (number, error), (heavy_number, _) in zip(light[signal][check], heavy[signal][check]):
                text = (f"{number:.0%}±{error:.0%} ({heavy_number:.0%})" if as_share
                        else f"{number:+.2f}±{error:.2f} ({heavy_number:+.2f})")
                cells += f"{text:>20s}"
            yield signal, cells

    print("Every number: the average over 10 runs ± its standard error, with 20 training steps per round;")
    print("in brackets, the same with 100 steps per round (the stress test, not part of any rule).\n")

    print("1. Ignores noise: share of toy pushes that are top picks (chance 25%, pass at most 30% everywhere).\n")
    print(f"{'':24s}{columns}")
    for signal, cells in table("toys among top picks"):
        print(f"  {signal:22s}{cells}   {mark[passes[signal][0]]}")
    print("\n   Beside it: share of rounds in which the toys' average score is above the other objects'.\n")
    for signal, cells in table("toys above average"):
        print(f"  {signal:22s}{cells}")

    print("\n2. Notices new things: share of arrivals after which a new object that is not a toy beats every old")
    print("   object, in the round it arrives or the next (pass at least 80% in both plain worlds; narrow ones")
    print("   for reference).\n")
    print(f"{'':24s}{columns}")
    for signal, cells in table("new object wins"):
        print(f"  {signal:22s}{cells}   {mark[passes[signal][1]]}")
    print("\n   Beside it: share of the new objects' pushes that are top picks in those two rounds (chance 25%).\n")
    for signal, cells in table("new among top picks"):
        print(f"  {signal:22s}{cells}")

    print("\n3. Finds rare useful pushes: chance that a narrow-part push outranks another push on the same kind of")
    print("   object (0.5 = no preference). Pass: at least 0.6 in both narrow worlds, and 0.05 above the plain world.")
    print("   Light training only. b = boxes, r = rollers, s = spinners.\n")
    print(f"{'':24s}" + "".join(f"{world + ': b r s':>20s}" for world in worlds))
    largest_error = 0
    for signal in signals:
        cells = ""
        for w in range(len(worlds)):
            numbers = [light[signal][f"narrow {kind}"][w] for kind in CHECKED_KINDS]
            largest_error = max(largest_error, *(error for _, error in numbers))
            cells += f"{' '.join(f'{number:.2f}' for number, _ in numbers):>20s}"
        print(f"  {signal:22s}{cells}   {mark[passes[signal][2]]}")
    print(f"   (standard errors at most {largest_error:.2f})")

    print("\nForecastable: how well the object's average score in the three rounds before forecasts the order of")
    print("each round's scores (1 = the same order, 0 = no relation). Not part of any rule.\n")
    print(f"{'':24s}{columns}")
    for signal, cells in table("forecastable", as_share=False):
        print(f"  {signal:22s}{cells}")

    print("\nPassed all three checks: " + (", ".join(s for s in signals if all(passes[s])) or "none"))


def print_noise_models(noise_models, worlds, separation):
    print("\n4. The noise models. First: chance that a toy push is rated noisier than a push on another object")
    print("   (pass at least 0.9). Then, for boxes, rollers and spinners: chance that a narrow-part push is rated")
    print("   noisier than another push on the same kind of object (pass at most 0.6). Only the narrow worlds")
    print("   count; plain ones for reference. Each line: 20 training steps per round; the line below it: 100.\n")
    print(f"{'':26s}" + "".join(f"{world + ': toys | b r s':>26s}" for world in worlds))
    for model in noise_models:
        for steps in (LIGHT, HEAVY):
            cells = ""
            for (toys, _), rare in separation[steps][model]:
                cells += f"{toys:.2f} | {' '.join(f'{number:.2f}' for number, _ in rare)}".rjust(26)
            if steps == LIGHT:
                narrow = [numbers for world, numbers in zip(worlds, separation[LIGHT][model]) if world.startswith("narrow")]
                ok = all(toys >= TOYS_NOISIER_AT_LEAST and all(number <= RARE_NOISIER_AT_MOST for number, _ in rare)
                         for (toys, _), rare in narrow)
                print(f"  {model:24s}{cells}   {'pass' if ok else 'FAIL'}")
            else:
                print(f"  {'  (100 steps)':24s}{cells}")


def print_guards(saved):
    budgets, worlds, guards = list(saved["budgets"]), list(saved["worlds"]), list(saved["guards"])
    end = saved["learnable"][..., -END_ROUNDS:].mean(axis=-1)  # [budget, world, guard, run]
    toys = saved["toy_error"][..., -END_ROUNDS:].mean(axis=-1)
    none = guards.index("none")
    change = end / end[:, :, none:none + 1] - 1
    print("\n5. The noise models as guards, with even pushing and 50 training steps per round. True error at the")
    print("   end on the learnable objects, then (in brackets) the change against no guard ± its standard error")
    print("   and in how many runs the guard did better, then the true error of the toys (memorized noise).\n")
    for b, budget in enumerate(budgets):
        print(f"{budget} budget" + "".join(f"{guard:>36s}" for guard in guards))
        for w, world in enumerate(worlds):
            cells = ""
            for g in range(len(guards)):
                runs = change[b, w, g]
                error = runs.std(ddof=1) / np.sqrt(len(runs))
                cells += (f"{end[b, w, g].mean():.4f} ({runs.mean():+4.0%}±{error:3.0%}, {(runs < 0).sum():2d}/{len(runs)})"
                          f" {toys[b, w, g].mean():.3f}").rjust(36)
            print(f"  {world:12s}{cells}")
        print()
    print("   The guard \"noise from repeats\" does every push twice under both budgets (its noise model needs the")
    print("   pairs), so its two rows are the same runs: 128 different pushes per round, not a second result.")


def draw(signals, worlds, light):
    """One panel per check, one row per signal, light training. The shaded side of each panel is where a signal passes."""
    figure, panels = plt.subplots(1, 3, figsize=(15, 0.36 * len(signals) + 1.8), sharey=True, facecolor=PAPER)
    rows = np.arange(len(signals))[::-1]  # the first signal at the top

    for panel, check, title, (low, high), chance in (
        (panels[0], "toys among top picks", "1. Ignores noise: toy pushes among the top picks", (0, TOYS_AT_MOST), TOP_SHARE),
        (panels[1], "new object wins", "2. Notices new things: a new object beats every old one", (NOTICED_AT_LEAST, 1), None),
    ):
        panel.axvspan(low, high, color=PASS_SHADE, zorder=0)
        if chance is not None:
            panel.axvline(chance, color=FAINT_INK, linewidth=1, linestyle="--")  # what a signal that knows nothing gives
        for w, world in enumerate(worlds):
            numbers = np.array([light[signal][check][w] for signal in signals])
            panel.errorbar(numbers[:, 0], rows + (1.5 - w) * 0.12, xerr=numbers[:, 1], fmt="o", markersize=4,
                           color=WORLD_COLOURS[world], elinewidth=1, zorder=3,
                           label=world.replace("narrow ", "narrow, ") + " objects")
        panel.set_xlim(-0.02, 1.02)
        panel.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        panel.set_title(title, loc="left", fontsize=10, color=INK)

    # Check 3: each kind's outrank chance in the narrow worlds (filled) and the plain worlds (hollow), averaged over sizes.
    panel = panels[2]
    panel.axvspan(RARE_AT_LEAST, 1, color=PASS_SHADE, zorder=0)
    panel.axvline(0.5, color=FAINT_INK, linewidth=1, linestyle="--")
    for k, kind in enumerate(CHECKED_KINDS):
        narrow = [np.mean([light[signal][f"narrow {kind}"][worlds.index(f"narrow {size}")][0] for size in ("6", "24")])
                  for signal in signals]
        plain = [np.mean([light[signal][f"narrow {kind}"][worlds.index(size)][0] for size in ("6", "24")])
                 for signal in signals]
        offset = rows + (1 - k) * 0.22  # the three kinds side by side in each row
        panel.hlines(offset, plain, narrow, color=KIND_COLOURS[k], linewidth=1, alpha=0.6)
        panel.scatter(plain, offset, s=22, facecolors=PAPER, edgecolors=KIND_COLOURS[k], linewidths=1.2, zorder=3)
        panel.scatter(narrow, offset, s=22, color=KIND_COLOURS[k], zorder=3, label=f"{kind}es" if kind == "box" else f"{kind}s")
    panel.set_xlim(0, 1)
    panel.set_title("3. Finds rare useful pushes: narrow parts outrank the rest", loc="left", fontsize=10, color=INK)

    for panel in panels:
        panel.set_yticks(rows, signals)
        tidy(panel)
        panel.grid(axis="x", color=GRID, linewidth=1)
    panels[0].legend(loc="upper left", bbox_to_anchor=(0, -0.04), ncol=2, frameon=False, fontsize=8)
    panels[2].legend(loc="upper left", bbox_to_anchor=(0, -0.04), ncol=3, frameon=False, fontsize=8,
                     title="filled: narrow worlds, hollow: plain worlds", title_fontsize=8)
    figure.suptitle("Step B, 20 training steps per round: each signal against the three checks "
                    "(shaded: pass; dashed: chance)", x=0.01, ha="left", fontsize=11, color=INK)
    figure.tight_layout()
    figure.savefig(RESULTS_FILE.replace(".npz", ".png"), dpi=160)


if __name__ == "__main__":
    main()
