"""Small checks that the self-play code does what it says.

Run them with:   ../.venv/bin/python -m unittest test_selfplay -v      (from the selfplay folder)
They run on the CPU. Two of them compare with the authors' released weights and are skipped if
those are not in data/selfplay/weights.
"""

import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import archive
import generator
import machine
import reward
import train
from model import INSTRUCTIONS, allowed_logits

CPU = torch.device("cpu")
RELEASED = Path(__file__).resolve().parent.parent / "data" / "selfplay" / "weights" / "100k" / "seed-40354564"
STEPS, TAPE = machine.limits(64, 1)


def random_programs(count, rng):
    symbols = list(INSTRUCTIONS + b"F")
    return [bytes([ord("S")] + list(rng.choice(symbols, size=rng.integers(1, 80)))) for _ in range(count)]


class MachineTests(unittest.TestCase):
    def test_fast_and_plain_machines_agree(self):
        rng = np.random.default_rng(0)
        programs = random_programs(400, rng)
        seeds = rng.integers(0, 2 ** 63, size=len(programs))
        outputs, written, depths = machine.run(programs, seeds, STEPS, TAPE)
        for i, program in enumerate(programs):
            output, count, depth = machine.reference_run(program, int(seeds[i]), STEPS, TAPE)
            self.assertTrue((output == outputs[i]).all(), program)
            self.assertEqual((count, depth), (written[i], depths[i]), program)

    def test_known_programs(self):
        cases = {   # program: (bytes written, loop depth)
            b"SX[.+]F": (240, 1),     # set 16, print and add 1 until the cell wraps to 0: 16..255
            b"S+[[-]]F": (0, 2),
            b"S[+]F": (0, 0),          # the cell is 0, so the loop is skipped and does not count
            b"S+WF": (0, 2),           # W is [[-]>+<], two loops deep once expanded
            b"S]+[F": (0, 0),          # brackets without partners do nothing
            b"S+[.]F": (4095, 1),      # prints 1 forever: stops when the output is full
            b"S+[]F": (0, 1),          # loops forever: stops at the step limit
        }
        outputs, written, depths = machine.run(list(cases), np.zeros(len(cases), dtype=np.int64), STEPS, TAPE)
        for i, (program, expected) in enumerate(cases.items()):
            self.assertEqual((written[i], depths[i]), expected, program)
        self.assertEqual(outputs[0, 1:241].tolist(), list(range(16, 256)))
        self.assertTrue((outputs[:, 0] == ord("O")).all())


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = train.new_model(train.Run(), CPU)

    def test_samples_are_well_formed(self):
        for program in generator.sample(self.model, 64, CPU):
            self.assertEqual(program[0], ord("S"))
            self.assertTrue(set(program[1:]) <= set(INSTRUCTIONS + b"F"))
            self.assertTrue(program.endswith(b"F") or len(program) == 1 + generator.MAX_PROGRAM)
            self.assertNotIn(ord("F"), program[1:-1])

    def test_log_probability_adds_up_the_sampling_steps(self):
        programs = generator.sample(self.model, 8, CPU)
        batched = generator.log_probs(self.model, programs, CPU)
        for program, total in zip(programs, batched.tolist()):
            stepwise = 0.0
            for t in range(1, len(program)):
                prefix = torch.tensor([list(program[:t])])
                logits = allowed_logits(self.model(prefix), prefix)[0, -1]
                stepwise += torch.log_softmax(logits, dim=-1)[program[t]].item()
            self.assertAlmostEqual(total, stepwise, places=3)

    def test_prior(self):
        self.assertAlmostEqual(generator.prior_log_probs([b"S+-F"])[0].item(), -3 * math.log(19), places=5)
        programs = generator.uniform_programs(20000, np.random.default_rng(0))
        lengths = np.array([len(p) - 1 for p in programs])     # symbols, F included: 19 on average under g0
        self.assertAlmostEqual(lengths.mean(), 19, delta=0.5)


class ArchiveTests(unittest.TestCase):
    def test_niches(self):
        self.assertEqual(archive.niche(b"S" + b"+" * 8 + b"F", 3), (3, 0))
        self.assertEqual(archive.niche(b"S" + b"+" * 9 + b"F", 12), (8, 1))    # deeper loops count as 8
        self.assertEqual(archive.niche(b"S" + b"+" * 17 + b"F", 0), (0, 2))
        self.assertEqual(archive.niche(b"S" + b"+" * 33 + b"F", 0), (0, 3))

    def test_admission_keeps_the_best_positive_programs(self):
        store = archive.Archive()
        programs = [b"S" + b"+" * n + b"F" for n in range(1, 13)]           # all in niche (0, 0) or (0, 1)
        store.admit(programs[:8], [0] * 8, [float(n) for n in range(8)])     # the first has reward 0: left out
        self.assertEqual(len(store), 7)
        store.admit([b"S-F", b"S<F"], [0, 0], [100.0, 0.5])                  # the niche is now over-full
        elites = store.niches[(0, 0)]
        self.assertEqual(len(elites), archive.ELITES)
        self.assertIn(b"S-F", elites)
        self.assertNotIn(b"S<F", elites)                                     # the lowest, so it was dropped
        store.decay()
        self.assertAlmostEqual(elites[b"S-F"], 100.0 * archive.DECAY)

    def test_parents_are_drawn_evenly_across_niches(self):
        store = archive.Archive()
        store.admit([b"S" + b"+" * n + b"F" for n in range(1, 9)], [0] * 8, [1.0] * 8)   # 8 in niche (0, 0)
        store.admit([b"S" + b"-" * 20 + b"F"], [0], [1.0])                                 # 1 in niche (0, 2)
        children = store.mutants(4000, np.random.default_rng(0))
        long_ones = np.mean([len(archive.body(c)) > 16 for c in children])
        self.assertAlmostEqual(long_ones, 0.5, delta=0.05)

    def test_a_mutant_is_one_edit_away(self):
        rng = np.random.default_rng(0)
        for parent in random_programs(300, rng):
            child = archive.mutate(parent, rng)
            a, b = archive.body(parent), archive.body(child)
            self.assertTrue(child.startswith(b"S") and child.endswith(b"F"))
            self.assertTrue(set(b) <= set(INSTRUCTIONS))
            self.assertEqual(edit_distance(a, b), 1, (parent, child))
        cut_off = b"S" + b"+" * generator.MAX_PROGRAM                     # 255 symbols and no F
        child = archive.mutate(cut_off, rng)
        self.assertEqual((len(archive.body(child)), child[-1]), (generator.MAX_PROGRAM - 1, ord("F")))

    def test_replay_draws_without_repeats(self):
        bank = archive.ReplayBank()
        bank.add([bytes([ord("S"), i, ord("F")]) for i in range(10)], range(10))
        programs, log_probs = bank.draw(8, np.random.default_rng(0))
        self.assertEqual(len(set(programs)), 8)
        self.assertEqual([p[1] for p in programs], [int(x) for x in log_probs])
        self.assertEqual(len(bank.draw(50, np.random.default_rng(0))[0]), 10)


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        previous, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] != b[j - 1]))
    return row[-1]


class RewardTests(unittest.TestCase):
    def test_reward_matches_the_exact_derivative(self):
        """After some real training, the reward is |<gradient of the loss, direction>| from autograd."""
        torch.manual_seed(0)
        learner = train.new_model(train.Run(), CPU)
        optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-2, betas=(0.9, train.Run().adam_beta2), weight_decay=0.0)
        rng = np.random.default_rng(0)
        start = reward.snapshot(learner)
        printing = [b"SX[.+]F", b"S,[.,]F", b"S++[.>+<]F", b"SX[>+.<-]F"]   # programs whose outputs differ
        for _ in range(6):
            programs = generator.uniform_programs(4, rng) + printing
            rows = torch.from_numpy(machine.run(programs, rng.integers(0, 2 ** 63, size=8), STEPS, TAPE)[0].astype(np.int64))
            train.learner_step(learner, optimizer, rows, CPU)
        along = reward.direction(learner, optimizer, start)
        ours = reward.alignment(learner, rows, along, CPU)
        exact = []
        for row in rows:
            learner.zero_grad()
            F.cross_entropy(learner(row[None, :-1])[0], row[1:]).backward()
            exact.append(abs(sum((p.grad * along[name]).sum() for name, p in learner.named_parameters()).item()))
        exact = torch.tensor(exact)
        self.assertGreater(len(set(exact.tolist())), 4)
        self.assertLess(((ours - exact).abs() / exact.max()).max().item(), 1e-4, (ours, exact))

    def test_no_movement_gives_no_reward(self):
        torch.manual_seed(0)
        learner = train.new_model(train.Run(), CPU)
        optimizer = torch.optim.AdamW(learner.parameters())
        rows = torch.zeros(2, 4096, dtype=torch.long)
        along = reward.direction(learner, optimizer, reward.snapshot(learner))
        self.assertEqual(reward.alignment(learner, rows, along, CPU).abs().max().item(), 0)

    def test_novelty_is_zero_for_an_output_seen_often_enough(self):
        torch.manual_seed(0)
        learner = train.new_model(train.Run(), CPU)
        rng = np.random.default_rng(0)
        printing = [b"SX[.+]F", b"S++[.>+<]F"]
        rows = torch.from_numpy(machine.run(printing, rng.integers(0, 2 ** 63, size=2), STEPS, TAPE)[0].astype(np.int64))
        seen = rows[:1].repeat(reward.NEAREST, 1)          # the first output, NEAREST times
        scores = reward.novelty(learner, rows, seen, CPU)
        self.assertEqual(scores[0].item(), 0)
        self.assertGreater(scores[1].item(), 0)

    def test_novelty_distances_are_exact_for_close_outputs(self):
        """40 outputs that differ in one byte have features far closer together than their length;
        the distances must still come out right (checked against 64-bit arithmetic)."""
        torch.manual_seed(0)
        learner = train.new_model(train.Run(), CPU)
        rng = np.random.default_rng(0)
        row = machine.run([b"SX[.+]F"], rng.integers(0, 2 ** 63, size=1), STEPS, TAPE)[0][0].astype(np.int64)
        rows = torch.from_numpy(np.tile(row, (40, 1)))
        rows[:, 4000] = torch.arange(40)                   # the only difference
        ours = reward.novelty(learner, rows, rows[:0], CPU)
        f = reward.features(learner, rows, CPU).double()
        distances = (f[:, None, :] - f[None, :, :]).pow(2).sum(-1).sqrt()
        distances.fill_diagonal_(float("inf"))
        exact = distances.topk(reward.NEAREST, dim=1, largest=False).values.mean(dim=1)
        self.assertGreater(exact.min().item(), 0)
        self.assertLess(((ours.double() - exact).abs() / exact).max().item(), 1e-3, (ours, exact))

    def test_seen_outputs_keep_an_even_sample_of_everything(self):
        memory = reward.SeenOutputs(2, size=4)
        rng = np.random.default_rng(0)
        for value in range(1, 4):                          # 6 rows into room for 4
            memory.add(np.full((2, 2), value, dtype=np.uint8), rng)
        self.assertEqual((memory.seen, memory.kept), (6, 4))
        self.assertEqual(len(memory.draw(10, rng)), 4)
        # Offer outputs 0 to 999 into room for 100: the ones kept should be spread over all of
        # them (mean about 500), not bunched at the end as a window of recent ones would be (950).
        means = []
        for trial in range(50):
            memory = reward.SeenOutputs(2, size=100)
            numbers = np.arange(1000)
            memory.add(np.stack([numbers // 256, numbers % 256], axis=1).astype(np.uint8), rng)
            kept = memory.rows[:, 0].astype(int) * 256 + memory.rows[:, 1]
            means.append(kept.mean())
        self.assertLess(abs(np.mean(means) - 499.5), 20, np.mean(means))


class TrainingTests(unittest.TestCase):
    def test_learning_rates_at_round_0_match_the_released_first_step(self):
        run = train.Run()
        self.assertAlmostEqual(train.learning_rate(0, run), 2.0e-4)
        self.assertAlmostEqual(train.learning_rate(0, run) * run.generator_ratio, 4.0e-5)
        self.assertAlmostEqual(train.learning_rate(run.warmup + 10, run), run.learning_rate)

    def test_mutants_do_not_break_the_generator_step(self):
        torch.manual_seed(0)
        model = train.new_model(train.Run(), CPU)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        programs = generator.sample(model, 6, CPU)
        kinds = np.array([train.FRESH, train.FRESH, train.MUTANT, train.MUTANT, train.REPLAY, train.REPLAY])
        sampled = np.array([np.nan, np.nan, np.nan, np.nan, -30.0, np.nan])   # the last was a mutant when stored
        train.generator_step(model, optimizer, programs, kinds, sampled, torch.rand(6), 0.01, CPU)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_stopping_and_continuing_gives_the_same_run(self):
        small = dict(fresh=4, mutants=4, replay=4)
        with tempfile.TemporaryDirectory() as folder:
            whole = train.train(train.Run(rounds=5, **small), Path(folder) / "whole", where=CPU, scores=False)
            train.train(train.Run(rounds=3, **small), Path(folder) / "parts", where=CPU, scores=False)
            parts = train.train(train.Run(rounds=5, **small), Path(folder) / "parts", where=CPU, scores=False)
            for key in whole["rounds"]:
                if key != "seconds":
                    self.assertEqual(whole["rounds"][key], parts["rounds"][key], key)
            a = torch.load(Path(folder) / "whole" / "generator_0004.pth")["generator_state_dict"]
            b = torch.load(Path(folder) / "parts" / "generator_0004.pth")["generator_state_dict"]
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_stopping_and_continuing_gives_the_same_run_for_our_arms(self):
        """Novelty also has to carry its memory of the outputs seen across the stop. The memory is
        made small here, so that it is full and replacing outputs by then."""
        for arm in ("novelty", "prediction_error"):
            small = dict(arm=arm, fresh=4, mutants=4, replay=4)
            with tempfile.TemporaryDirectory() as folder, mock.patch.object(reward, "MEMORY", 20):
                whole = train.train(train.Run(rounds=5, **small), Path(folder) / "whole", where=CPU, scores=False)
                train.train(train.Run(rounds=3, **small), Path(folder) / "parts", where=CPU, scores=False)
                parts = train.train(train.Run(rounds=5, **small), Path(folder) / "parts", where=CPU, scores=False)
                for key in whole["rounds"]:
                    if key != "seconds":
                        self.assertEqual(whole["rounds"][key], parts["rounds"][key], (arm, key))
                self.assertGreater(len(set(whole["rounds"]["reward_mean"])), 1, arm)

    @unittest.skipUnless((RELEASED / "learner_0000.pth").exists(), "released weights not downloaded")
    def test_starting_weights_look_like_the_released_ones(self):
        released = torch.load(RELEASED / "learner_0000.pth", map_location="cpu", weights_only=True)["learner_state_dict"]
        ours = train.new_model(train.Run(), CPU).state_dict()
        for name, theirs in released.items():
            if theirs.is_floating_point() and "norm" not in name and "ln_f" not in name:
                self.assertAlmostEqual(ours[name].std().item() / theirs.std().item(), 1, delta=0.05, msg=name)

    @unittest.skipUnless((RELEASED / "learner_0000.pth").exists(), "released weights not downloaded")
    def test_first_step_moves_norm_weights_as_much_as_the_released_one(self):
        torch.manual_seed(0)
        run = train.Run()
        learner = train.new_model(run, CPU)
        optimizer = torch.optim.AdamW(learner.parameters(), lr=train.learning_rate(0, run), betas=(0.9, train.Run().adam_beta2),
                                      weight_decay=0.0)
        rows = torch.from_numpy(machine.run(generator.uniform_programs(8, np.random.default_rng(0)),
                                            np.arange(8), STEPS, TAPE)[0].astype(np.int64))
        train.learner_step(learner, optimizer, rows, CPU)
        released = torch.load(RELEASED / "learner_0000.pth", map_location="cpu", weights_only=True)["learner_state_dict"]
        name = "blocks.0.input_layernorm.weight"
        self.assertAlmostEqual((learner.state_dict()[name] - 1).abs().median().item(),
                               (released[name] - 1).abs().median().item(), delta=2e-6)


if __name__ == "__main__":
    unittest.main()
