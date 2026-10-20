"""The test set: steps from all over the game, to measure how well a world model knows it.

It is built once, from separate long runs of two explorers that never learn: cell counts and
uniform choice, two runs each, with seeds no learner uses. It never comes from a learner's own
run, so it cannot favour one. From the places these runs saved, up to PLACES_PER_ROOM are taken
in every room they reached, and from each, every one of the 18 buttons is pressed once. The
emulator tells us what really happens: the true change of every byte.

How well a world model knows the game (TestSet.knowledge), every room counting the same:
    1. its error in each room: for each test step and byte, the squared miss of the change it
       guesses, divided by how much that byte moves in the test set (its squared change,
       averaged within each room and then over the rooms, but at least 1, as in training: a
       change of one byte is the smallest there is), then averaged over the bytes and the
       room's steps. Dividing by each byte's movement matters: bytes 5 and 6 jump around like
       a random number generator, and would otherwise be most of the error. The floor of 1
       matters too: some bytes change in a few dozen of the 15,000 test steps, and without it
       the smallest leftover guess on them would count hundreds of times over. Bytes that never
       change in the test set are left out: they say nothing about how the game moves.
    2. its knowledge of each room: 1 - its error there / the error of guessing "nothing moves"
       there. 1 means it predicts every move; 0 means it does no better than "nothing moves".
       A room it predicts worse than that counts as 0: in a room it has never seen, a network
       guesses wildly, and that is knowing nothing, not knowing less than nothing.
    3. its knowledge of the game: the average over the rooms.

Run it with:   .venv/bin/python test_set.py     (about 20 minutes; writes results/montezuma_test_set.npz)
"""

from multiprocessing import Pool

import numpy as np

from explore import rounds, start
from game import BUTTONS, ROOM_BYTE, change
from learners import LEARNERS

TEST_SET_FILE = "results/montezuma_test_set.npz"
EXPLORERS = (("cell counts", 101), ("cell counts", 102), ("uniform choice", 103), ("uniform choice", 104))
EXPLORER_FRAMES = 16_000_000
PLACES_PER_ROOM = 100


class TestSet:
    def __init__(self, before, button, after):
        self.before, self.button, self.after = before, button, after
        self.room = before[:, ROOM_BYTE]
        self.rooms = np.unique(self.room)
        true_change = change(before, after)
        self.moving = (true_change != 0).any(axis=0)   # bytes that change at least once
        self.true_change = true_change[:, self.moving]
        self.no_change_misses = self.room_squared_misses(np.zeros_like(self.true_change))  # of "nothing moves"
        self.movement = np.maximum(self.no_change_misses.mean(axis=0), 1)  # averaged over rooms, at least 1
        self.no_change = self.room_errors(self.no_change_misses)          # the error of "nothing moves", per room

    def __len__(self):
        return len(self.button)

    def room_squared_misses(self, guessed_change):
        """[room, moving byte]: the squared miss of these guessed changes (one row per test step), averaged in each room."""
        squared = (guessed_change - self.true_change).astype(float) ** 2
        return np.array([squared[self.room == room].mean(axis=0) for room in self.rooms])

    def room_errors(self, squared_misses):
        """The error in each room: each byte's squared miss divided by its movement, averaged over the bytes."""
        return (squared_misses / self.movement).mean(axis=-1)

    def knowledge(self, model):
        """How well the world model knows the game, from 0 to 1 (see the top of this file).

        Also returns its error in each room, and its squared miss in each room and byte, so that
        a report can weigh the bytes another way without running the learner again.
        """
        squared_misses = self.room_squared_misses(model.guess(self.before, self.button)[:, self.moving])
        room_errors = self.room_errors(squared_misses)
        return knowledge(room_errors, self.no_change), room_errors, squared_misses


def knowledge(room_errors, no_change):
    """The average over rooms of 1 - error / error of "nothing moves", each room at least 0. Works on [..., room]."""
    return np.maximum(1 - room_errors / no_change, 0).mean(axis=-1)


def load():
    saved = np.load(TEST_SET_FILE)
    return TestSet(saved["before"], saved["button"], saved["after"])


def explore_and_press(rule, seed, places_per_room=PLACES_PER_ROOM):
    """Explore without learning, then press every button once from up to places_per_room places per room."""
    rng = np.random.default_rng(seed)
    game, places = start(seed)
    explorer = LEARNERS[rule](rng, None, None)  # it chooses, but has no world model to learn
    for _ in rounds(explorer, game, places, rng, EXPLORER_FRAMES):
        pass
    rooms = np.array([memory[ROOM_BYTE] for memory in places.memory])
    picked = np.concatenate([rng.permutation(np.flatnonzero(rooms == room))[:places_per_room]
                             for room in np.unique(rooms)])
    before, button, after = [], [], []
    for place in picked:
        for b in range(BUTTONS):
            game.restore(places.saved[place])
            before.append(game.memory())
            game.press(b)
            after.append(game.memory())
            button.append(b)
    return np.array(before), np.array(button, dtype=np.uint8), np.array(after)


def build():
    """Run the explorers side by side, pool their steps, and keep up to PLACES_PER_ROOM places per room."""
    with Pool(len(EXPLORERS)) as pool:
        parts = pool.starmap(explore_and_press, EXPLORERS)
    before, button, after = (np.concatenate(part) for part in zip(*parts))
    # Pooled over the explorers, a room can have up to 4 x PLACES_PER_ROOM places. Keep PLACES_PER_ROOM,
    # each with its 18 presses together, so that every room is tested on at most the same number of places.
    rng = np.random.default_rng(0)
    place = np.arange(len(button)) // BUTTONS
    room = before[::BUTTONS, ROOM_BYTE]
    keep = np.concatenate([rng.permutation(np.flatnonzero(room == r))[:PLACES_PER_ROOM] for r in np.unique(room)])
    kept = np.isin(place, keep)
    np.savez_compressed(TEST_SET_FILE, before=before[kept], button=button[kept], after=after[kept],
                        explorers=[f"{rule} {seed}" for rule, seed in EXPLORERS], frames=EXPLORER_FRAMES)
    test_set = load()
    print(f"{len(test_set)} test steps in {len(test_set.rooms)} rooms: {test_set.rooms.tolist()}")
    print(f"places per room: {[int((room[keep] == r).sum()) for r in test_set.rooms]}")
    print(f"bytes that move: {int(test_set.moving.sum())} of 128")


if __name__ == "__main__":
    build()
