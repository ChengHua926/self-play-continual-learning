"""Small checks that the game harness does what it says.

Run them with:   .venv/bin/python -m unittest test_game -v
"""

import unittest
from unittest.mock import patch

import numpy as np

import explore
from explore import LEFT_TO_CHANCE, PLACES_PER_ROUND, Places, choose, press_randomly, rounds, start
from learners import CONTROLS, LEARNERS
from game import BUTTONS, MEMORY_SIZE, ROOM_BYTE, Game, change
from test_set import TestSet, knowledge
from world_model import Replay, WorldModel, inputs


class GameTests(unittest.TestCase):
    def test_jumping_back_and_pressing_the_same_buttons_gives_the_same_memory(self):
        game = Game(seed=0)
        rng = np.random.default_rng(0)
        for button in rng.integers(BUTTONS, size=200):
            game.press(button)
        saved, presses = game.save(), rng.integers(BUTTONS, size=100)
        for button in presses:
            game.press(button)
        first = game.memory()
        game.restore(saved)
        for button in presses:
            game.press(button)
        np.testing.assert_array_equal(game.memory(), first)

    def test_a_byte_change_is_measured_the_short_way_round(self):
        before = np.array([[255, 0, 10, 200]], dtype=np.uint8)
        after = np.array([[0, 255, 13, 72]], dtype=np.uint8)
        np.testing.assert_array_equal(change(before, after), [[1, -1, 3, -128]])

    def test_frames_are_counted_four_per_press(self):
        game = Game(seed=0)
        for _ in range(10):
            game.press(0)
        self.assertEqual(game.frames_played, 40)


class ScriptedGame:
    """Stands in for the game: the screen code met after each press, and the lives left after each press."""

    def __init__(self, codes, lives):
        self.codes, self.lives_after = codes, lives  # lives_after[k]: lives left after k presses
        self.presses, self.frames_played = 0, 0

    def press(self, button):
        self.presses += 1
        self.frames_played += 4

    def memory(self):
        return np.full(MEMORY_SIZE, self.presses, dtype=np.uint8)

    def save(self):
        return self.presses

    def lives(self):
        return self.lives_after[self.presses]

    def over(self):
        return False

    def screen_code(self):
        return self.codes[self.presses - 1]


class PlacesTests(unittest.TestCase):
    def test_a_place_met_again_is_counted_not_saved_again(self):
        game, places = ScriptedGame(["a", "b", "a", "b", "a"] + ["new"] * 30, [5] * 40), Places()
        press_randomly(game, np.random.default_rng(0), places, presses=5, stop_at_lost_life=True)
        self.assertEqual(places.index, {"a": 0, "b": 1})
        self.assertEqual(places.visits, [3, 2])

    def test_a_place_keeps_the_saved_game_from_when_it_was_first_met(self):
        game, places = ScriptedGame(["a", "b", "a"] + ["new"] * 30, [5] * 40), Places()
        press_randomly(game, np.random.default_rng(0), places, presses=3, stop_at_lost_life=True)
        self.assertEqual(places.saved[places.index["a"]], 1)  # saved after the first press, not the third

    def test_places_met_in_the_last_20_presses_before_a_lost_life_are_not_saved(self):
        codes = [f"place {press}" for press in range(30)]
        lives = [5] * 30 + [4]  # the life is lost on press 29 (the 30th)
        game, places = ScriptedGame(codes, lives), Places()
        press_randomly(game, np.random.default_rng(0), places, presses=50, stop_at_lost_life=True)
        # Met on presses 0 to 8 (21 or more presses before the death): saved. Met on presses 9 to 28: not.
        self.assertEqual(list(places.index), codes[:9])
        self.assertEqual(game.presses, 30)  # and the presses stop at the lost life

    def test_when_the_presses_run_out_pressing_goes_on_until_the_last_new_places_are_safe(self):
        codes = [f"place {press}" for press in range(10)]
        game, places = ScriptedGame(codes + ["met while waiting"] * 30, [5] * 41), Places()
        press_randomly(game, np.random.default_rng(0), places, presses=10, stop_at_lost_life=True)
        self.assertEqual(list(places.index), codes)  # the code met while waiting is not saved
        self.assertEqual(game.presses, 30)          # "place 9" was met on press 9 and waited 20 more

    def test_a_life_lost_while_waiting_drops_the_waiting_places(self):
        codes = [f"place {press}" for press in range(10)]
        lives = [5] * 16 + [4] * 30  # the life is lost on press 15, after the 10 presses ran out
        game, places = ScriptedGame(codes + ["falling"] * 30, lives), Places()
        press_randomly(game, np.random.default_rng(0), places, presses=10, stop_at_lost_life=True)
        self.assertEqual(len(places), 0)
        self.assertEqual(game.presses, 16)

    def test_random_play_goes_on_after_a_lost_life(self):
        codes = [f"place {press}" for press in range(60)]
        lives = [5] * 6 + [4] * 60  # a life is lost on press 5
        game, places = ScriptedGame(codes, lives), Places()
        press_randomly(game, np.random.default_rng(0), places, presses=40, stop_at_lost_life=False)
        self.assertGreaterEqual(game.presses, 40)
        self.assertIn("place 6", places.index)  # met after the lost life, more than 20 presses before the end

    def test_every_run_of_presses_is_credited_with_the_new_places_it_met(self):
        game, places = start(seed=0)
        rng = np.random.default_rng(0)
        bursts = next(rounds(LEARNERS["uniform choice"](rng, None, None), game, places, rng, frames=10**9))
        self.assertEqual(sum(places.chosen), PLACES_PER_ROUND)  # each jump back counted
        saved_now = set(places.index) - {next(iter(places.index))}
        for burst in bursts:
            self.assertEqual(burst.new_places, len(burst.met & saved_now))
        # All 32 started from the one place there was, so several met the same new places, and each is credited.
        self.assertGreater(sum(burst.new_places for burst in bursts), len(saved_now))

    def test_a_saved_place_brings_back_its_memory(self):
        game, places = start(seed=0)
        rng = np.random.default_rng(1)
        press_randomly(game, rng, places, presses=300, stop_at_lost_life=False)
        self.assertGreater(len(places), 1)
        for place in range(len(places)):
            game.restore(places.saved[place])
            np.testing.assert_array_equal(game.memory(), places.memory[place])


class ChoosingTests(unittest.TestCase):
    def setUp(self):
        self.places = Places()
        self.places.saved = [None] * 100
        self.places.visits = list(range(1, 101))  # place 0 was visited least
        self.places.chosen = [0] * 100

    def test_a_round_jumps_back_to_32_places_a_quarter_of_them_by_chance(self):
        rng = np.random.default_rng(0)
        chosen = choose(LEARNERS["uniform choice"](rng, None, None), self.places, rng)
        self.assertEqual(len(chosen), PLACES_PER_ROUND)
        self.assertEqual(round(LEFT_TO_CHANCE * PLACES_PER_ROUND), 8)

    def test_cell_counts_prefers_places_jumped_back_to_least_then_visited_least(self):
        places = Places()
        places.visits, places.chosen = [1, 1, 50], [0, 10, 0]
        score = LEARNERS["cell counts"](None, None, None).score(places, np.arange(3))
        # 0.3 / sqrt(visits) + 0.1 / sqrt(chosen), both with 0.001 added: never jumped back to counts most.
        self.assertEqual(list(np.argsort(-score)), [0, 2, 1])

    def test_the_controls_score_as_they_say(self):
        places = Places()
        places.visits = [1, 100, 50]
        places.memory = [np.full(MEMORY_SIZE, room, dtype=np.uint8) for room in (1, 1, 2)]  # rooms 1, 1, 2
        most_visited = CONTROLS["most visited"](None, None, None).score(places, np.arange(3))
        room_oracle = CONTROLS["room oracle"](None, None, None).score(places, np.arange(3))
        self.assertEqual(int(most_visited.argmax()), 1)
        # Place 0 is the least visited place, but its room has 101 visits; place 2's room has 50.
        self.assertEqual(int(room_oracle.argmax()), 2)

    def test_cell_counts_picks_the_least_visited_of_its_candidates(self):
        rng = np.random.default_rng(0)
        candidates = np.random.default_rng(0).integers(100, size=8 * 24)
        chosen = choose(LEARNERS["cell counts"](rng, None, None), self.places, rng)[:24]
        np.testing.assert_array_equal(np.sort(chosen), np.sort(candidates)[:24])


class PressingTests(unittest.TestCase):
    def test_a_run_of_presses_stops_right_after_a_life_is_lost(self):
        game, places = start(seed=0)
        rng = np.random.default_rng(0)
        saved = game.save()
        for _ in range(200):
            game.restore(saved)
            before, buttons, after, _ = press_randomly(game, rng, places, presses=50, stop_at_lost_life=True)
            self.assertEqual(before.shape, after.shape)
            self.assertEqual(len(buttons), len(before))
            lives_left = after[:, 58]  # memory byte 58 holds the lives left, less one
            if lives_left[-1] < before[0, 58]:
                # The life was lost on the last press, and on no earlier one.
                self.assertTrue((lives_left[:-1] == before[0, 58]).all())
                return
            self.assertEqual(len(buttons), 50)
        self.fail("no life was lost in 200 runs of presses")

    def test_random_play_starts_the_game_again_when_it_is_over(self):
        game, places = start(seed=0)
        rng = np.random.default_rng(0)
        press_randomly(game, rng, places, presses=20_000, stop_at_lost_life=False)
        self.assertFalse(game.over())

    def test_the_frame_budget_is_kept_and_random_play_never_jumps_back(self):
        game, places = start(seed=0)
        with patch.object(Game, "restore", side_effect=AssertionError("random play jumped back")):
            rng = np.random.default_rng(0)
            for _ in rounds(LEARNERS["random play"](rng, None, None), game, places, rng, frames=20_000):
                pass
        self.assertGreaterEqual(game.frames_played, 20_000)
        self.assertLess(game.frames_played, 20_000 + PLACES_PER_ROUND * explore.PRESSES * 4)


class WorldModelTests(unittest.TestCase):
    def test_inputs_have_the_memory_scaled_and_one_button_switched_on(self):
        memory = np.full((3, MEMORY_SIZE), 255, dtype=np.uint8)
        x = inputs(memory, np.array([0, 5, 17]))
        self.assertEqual(x.shape, (3, MEMORY_SIZE + BUTTONS))
        np.testing.assert_array_equal(x[:, :MEMORY_SIZE], 1.0)
        np.testing.assert_array_equal(x[:, MEMORY_SIZE:].sum(axis=1), 1)
        self.assertEqual(x[1, MEMORY_SIZE + 5], 1)

    def test_each_bytes_typical_change_is_its_root_mean_square_over_every_step(self):
        replay = Replay(np.random.default_rng(0), size=10)  # keeps only 10, but counts all 100
        before = np.zeros((100, MEMORY_SIZE), dtype=np.uint8)
        after = before.copy()
        after[:50, 0] = 4      # byte 0 moves by 4 in half the steps
        after[:, 1] = 255      # byte 1 moves by -1 in every step, the short way round
        replay.add(before, np.zeros(100, dtype=np.uint8), after)
        scale = replay.typical_change()
        self.assertAlmostEqual(scale[0], np.sqrt(8))  # sqrt((50 x 16) / 100)
        self.assertAlmostEqual(scale[1], 1.0)
        self.assertEqual(scale[2], 1.0)                # never moves: at least 1

    def test_the_replay_keeps_an_even_sample_of_every_step(self):
        rng = np.random.default_rng(0)
        replay = Replay(rng, size=1_000)
        steps = 20_000
        ids = np.arange(steps)
        memory = np.zeros((steps, MEMORY_SIZE), dtype=np.uint8)
        memory[:, 0], memory[:, 1] = ids % 256, ids // 256  # each step's number, written into its memory
        replay.add(memory, np.zeros(steps, dtype=np.uint8), memory)
        kept = replay.before[:, 0].astype(int) + 256 * replay.before[:, 1].astype(int)
        self.assertEqual((replay.kept, replay.seen), (1_000, steps))
        self.assertEqual(len(np.unique(kept)), 1_000)
        self.assertAlmostEqual(kept.mean() / steps, 0.5, delta=0.03)  # early and late steps alike

    def test_the_world_model_learns_the_steps_it_trains_on(self):
        game, places = start(seed=0)
        rng = np.random.default_rng(0)
        before, buttons, after, _ = press_randomly(game, rng, places, presses=3_000, stop_at_lost_life=False)
        replay = Replay(rng)
        replay.add(before, buttons, after)
        model = WorldModel(rng)
        test_set = TestSet(before, buttons.astype(np.uint8), after)  # scored on the very steps it trains on
        self.assertEqual(test_set.knowledge(model)[0], 0.0)  # untrained, it guesses wildly
        model.learn(replay, steps=300)
        after_300 = test_set.knowledge(model)[0]
        model.learn(replay, steps=1_000)
        self.assertGreater(after_300, 0.1)
        self.assertGreater(test_set.knowledge(model)[0], after_300)  # and it keeps learning
        # Each byte's typical change was measured on the replay, at least 1 byte.
        self.assertTrue((model.scale >= 1).all())
        self.assertGreater(model.scale.max(), 10)


class TestSetTests(unittest.TestCase):
    def make(self):
        rng = np.random.default_rng(0)
        before = rng.integers(256, size=(6, MEMORY_SIZE)).astype(np.uint8)
        before[:, ROOM_BYTE] = [1, 1, 1, 1, 2, 2]
        after = before.copy()
        after[:, 10] += 3              # one byte moves by 3 in every step
        after[:, 20] += np.array([0, 0, 0, 0, 100, 100], dtype=np.uint8)  # another moves a lot, in room 2 only
        after[:, 30] += np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)      # a third moves once, by 1
        return TestSet(before, np.zeros(6, dtype=np.uint8), after)

    def test_bytes_that_never_move_are_left_out(self):
        self.assertEqual(int(self.make().moving.sum()), 3)

    def test_a_byte_that_barely_moves_counts_as_moving_at_least_1(self):
        test_set = self.make()
        # Byte 30's squared change averages 1/4 in room 1 and 0 in room 2: 1/8 over rooms, floored to 1.
        np.testing.assert_allclose(test_set.movement, [9.0, 100**2 / 2, 1.0])

    def test_every_byte_and_every_room_counts_the_same(self):
        test_set = self.make()

        class NoChange:  # guesses that nothing moves
            def guess(self, memories, buttons):
                return np.zeros((len(memories), MEMORY_SIZE))

        knowledge, room_errors, squared_misses = test_set.knowledge(NoChange())
        # Byte 10 scores 1 everywhere. Byte 20 moves only in room 2: its squared change averages 100^2
        # there and 0 in room 1, so its movement is 100^2 / 2, and it scores 0 in room 1 and 2 in room 2.
        # Byte 30 scores 1/4 in room 1 and 0 in room 2. Each room averages its three bytes.
        # "Nothing moves" knows nothing: knowledge 0.
        np.testing.assert_allclose(room_errors, [(1 + 0 + 0.25) / 3, (1 + 2 + 0) / 3])
        np.testing.assert_allclose(test_set.no_change, room_errors)
        np.testing.assert_allclose(squared_misses[1], [9.0, 100**2, 0.0])  # room 2, in bytes squared
        self.assertAlmostEqual(knowledge, 0.0)

        class Exact:  # guesses every change right
            def guess(self, memories, buttons):
                return change(memories, test_set.after).astype(float)

        self.assertAlmostEqual(test_set.knowledge(Exact())[0], 1.0)

    def test_a_room_guessed_worse_than_nothing_moves_counts_as_knowing_nothing(self):
        # Room errors against "nothing moves" errors of [0.5, 1.5]: halving room 1's error is knowledge 0.5
        # there; tripling room 2's is knowledge 0 there, not -2. The game: the average, 0.25.
        self.assertAlmostEqual(knowledge(np.array([0.25, 4.5]), np.array([0.5, 1.5])), 0.25)


class RunTests(unittest.TestCase):
    def test_a_short_run_of_every_learner_measures_every_line(self):
        game, places = start(seed=0)
        before, buttons, after, _ = press_randomly(game, np.random.default_rng(0), places, 2_000, stop_at_lost_life=False)
        small_test_set = TestSet(before[:500], buttons[:500].astype(np.uint8), after[:500])
        with patch("explore.MEASURE_EVERY", 10_000):
            for learner in ("uniform choice", "cell counts", "random play"):
                lines = explore.run(learner, 0, small_test_set, frames=30_000)
                self.assertEqual(len(lines["frames"]), 3, learner)
                self.assertTrue(((lines["knowledge"] >= 0) & (lines["knowledge"] <= 1)).all(), learner)
                self.assertGreater(lines["training_steps"][-1], 0, learner)
                self.assertTrue((np.diff(lines["places"]) >= 0).all(), learner)


if __name__ == "__main__":
    unittest.main()
