"""The push worlds: objects on a table, and one thing you can do to them: push.

Nothing here is real physics. The rules are simple on purpose. There are four kinds of
object, with easy rules, harder rules, and one kind that moves by pure chance:

    box       slides straight along the push. Boxes differ in how far they slide.
    roller    rolls far the way it is facing, and hardly at all sideways.
    spinner   hit it off-centre and it turns instead of sliding.
    toy       ignores the push and moves at random. This is the noise.

TABLE is the world of steps 1 and 2: six objects, two of which arrive late. Steps 1 and 2
use it through the plain names at the bottom of this file (OBJECTS, random_pushes, push,
and so on). make_world builds bigger worlds of the same four kinds, for the map.

A world can also be narrow. Then each object shows its own behaviour only in a narrow part
of the pushes, so most random pushes teach little:

    box       moves only when pushed hard (strength 0.8 or more). Otherwise it stays put.
    roller    rolls only when pushed within 15 degrees of the way it faces. It still scrapes
              a little sideways.
    spinner   turns only when hit near its edge (0.85 or more off-centre). It still slides.
    toy       the same as always.
"""

from dataclasses import dataclass, replace

import numpy as np

RESULT_SIZE = 3  # every push ends in 3 numbers: moved along x, moved along y, turned

# The narrow parts of a narrow world.
STRONG = 0.8               # a box moves only when pushed at least this hard
ALIGNED = np.radians(15)   # a roller rolls only when pushed within this angle of the way it faces
EDGE = 0.85                # a spinner turns only when hit at least this far off-centre


@dataclass
class Pushes:
    """A batch of pushes. Every field is an array with one entry per push."""

    obj: np.ndarray        # which object, as an index into the world's objects
    direction: np.ndarray  # which way the push goes, as an angle in radians
    strength: np.ndarray   # from 0.2 (gentle) to 1.0 (hard)
    hit_point: np.ndarray  # from -1 to 1: where along the object's side the push lands
    facing: np.ndarray     # which way the object is facing, as an angle in radians

    def __len__(self):
        return len(self.obj)

    def take(self, which):
        """The pushes picked out by which: a list of positions, or a yes/no per push."""
        return Pushes(**{name: values[which] for name, values in vars(self).items()})


def joined(*batches):
    """Several batches of pushes as one batch."""
    return Pushes(**{name: np.concatenate([vars(batch)[name] for batch in batches]) for name in vars(batches[0])})


@dataclass
class World:
    """The objects of one world. Every field has one entry per object."""

    names: tuple            # what each object is called
    kinds: tuple            # "box", "roller", "spinner" or "toy"
    reach: np.ndarray       # box, roller, spinner: how far a full-strength push moves it. Toy: its largest random move.
    twist: np.ndarray       # roller: how far it scrapes sideways. Spinner: how much it turns. Toy: its largest random turn.
    appears_at: np.ndarray  # the round it is first put on the table
    narrow: bool = False    # whether each object shows its behaviour only in a narrow part of the pushes

    @property
    def input_size(self):
        """The network reads one on/off switch per object, plus 6 numbers about the push."""
        return len(self.names) + 6

    def random_pushes(self, count, shares, rng):
        """Draw pushes at random.

        shares[i] says how often object i gets picked. A share of 0 means that object
        is not on the table.
        """
        shares = np.asarray(shares, dtype=float)
        return Pushes(
            obj=rng.choice(len(self.names), size=count, p=shares / shares.sum()),
            direction=rng.uniform(0, 2 * np.pi, count),
            strength=rng.uniform(0.2, 1.0, count),
            hit_point=rng.uniform(-1, 1, count),
            facing=rng.uniform(0, 2 * np.pi, count),
        )

    def expected_result(self, pushes):
        """What the rules say each push does: (moved along x, moved along y, turned).

        For a toy this is all zeros. A toy moves by chance, so zero is its average
        result, and the average is the best guess anyone can make.
        """
        result = np.zeros((len(pushes), RESULT_SIZE))
        moved = result[:, :2]   # writing into these two also writes into result
        turned = result[:, 2]

        kind = np.asarray(self.kinds)[pushes.obj]
        reach = self.reach[pushes.obj][:, None]
        twist = self.twist[pushes.obj]
        power = (pushes.strength**2)[:, None]
        along_push = _unit_vector(pushes.direction)
        along_facing = _unit_vector(pushes.facing)
        across_facing = _unit_vector(pushes.facing + np.pi / 2)

        box = kind == "box"
        if self.narrow:
            box &= pushes.strength >= STRONG
        moved[box] = (reach * power * along_push)[box]

        roller = kind == "roller"
        angle = (pushes.direction - pushes.facing)[:, None]
        rolled = reach * power * np.cos(angle) * along_facing
        if self.narrow:
            rolled *= np.abs(np.sin(angle)) <= np.sin(ALIGNED)
        scraped = twist[:, None] * power * np.sin(angle) * across_facing
        moved[roller] = (rolled + scraped)[roller]

        spinner = kind == "spinner"
        off_centre = np.abs(pushes.hit_point)[:, None]
        moved[spinner] = (reach * power * (1 - off_centre) * along_push)[spinner]
        turns = twist * pushes.strength * pushes.hit_point
        if self.narrow:
            turns *= np.abs(pushes.hit_point) >= EDGE
        turned[spinner] = turns[spinner]

        return result

    def in_narrow_part(self, pushes):
        """yes/no per push: does it land in its object's narrow part?

        The narrow parts are where a narrow world's objects show their behaviour: a box
        pushed hard, a roller pushed along the way it faces (or straight against it), a
        spinner hit near its edge. Toys have none. A plain world has the same parts, but
        nothing special happens in them.
        """
        kind = np.asarray(self.kinds)[pushes.obj]
        along_facing = np.abs(np.sin(pushes.direction - pushes.facing)) <= np.sin(ALIGNED)
        return (((kind == "box") & (pushes.strength >= STRONG))
                | ((kind == "roller") & along_facing)
                | ((kind == "spinner") & (np.abs(pushes.hit_point) >= EDGE)))

    def test_pushes(self, count, rng):
        """Pushes for measuring the true error, every object equally often.

        In a narrow world, about half of them are moved into each object's narrow part, so
        that knowing an object's rare behaviour counts as much as knowing its common one.
        """
        pushes = self.random_pushes(count, np.ones(len(self.names)), rng)
        if not self.narrow:
            return pushes
        kind = np.asarray(self.kinds)[pushes.obj]
        narrow_half = rng.random(count) < 0.5

        box = narrow_half & (kind == "box")
        pushes.strength[box] = rng.uniform(STRONG, 1.0, box.sum())

        roller = narrow_half & (kind == "roller")
        towards_or_away = rng.integers(0, 2, roller.sum()) * np.pi
        pushes.direction[roller] = pushes.facing[roller] + towards_or_away + rng.uniform(-ALIGNED, ALIGNED, roller.sum())

        spinner = narrow_half & (kind == "spinner")
        pushes.hit_point[spinner] = rng.choice([-1, 1], spinner.sum()) * rng.uniform(EDGE, 1.0, spinner.sum())
        return pushes

    def push(self, pushes, rng):
        """Do the pushes and return what happened.

        Every object follows its rule exactly, except a toy, which ignores the push.
        """
        result = self.expected_result(pushes)
        toy = np.asarray(self.kinds)[pushes.obj] == "toy"
        toys = pushes.obj[toy]
        count = toy.sum()
        distance = self.reach[toys][:, None] * rng.uniform(0, 1, count)[:, None]
        result[toy, :2] = distance * _unit_vector(rng.uniform(0, 2 * np.pi, count))
        result[toy, 2] = self.twist[toys] * rng.uniform(-1, 1, count)
        return result

    def as_inputs(self, pushes):
        """Turn pushes into the numbers the network reads.

        One on/off switch per object says which object it is. Angles go in as (cos, sin)
        so that 359 degrees and 1 degree look close to each other.
        """
        first = len(self.names)  # where the numbers about the push start
        inputs = np.zeros((len(pushes), self.input_size))
        inputs[np.arange(len(pushes)), pushes.obj] = 1
        inputs[:, first:first + 2] = _unit_vector(pushes.direction)
        inputs[:, first + 2] = pushes.strength
        inputs[:, first + 3] = pushes.hit_point
        inputs[:, first + 4:first + 6] = _unit_vector(pushes.facing)
        return inputs


def near_copies(pushes, nudge, rng, copies):
    """copies pushes close to each push, one push's copies after the other.

    Each copy has the same object, and every setting moved at random by up to nudge of its
    range (direction and facing by up to nudge of half a turn, strength and hit point by up
    to nudge of their span, kept inside their limits).
    """
    count = len(pushes) * copies
    copy = pushes.take(np.repeat(np.arange(len(pushes)), copies))
    copy.direction = copy.direction + nudge * np.pi * rng.uniform(-1, 1, count)
    copy.strength = np.clip(copy.strength + nudge * 0.8 * rng.uniform(-1, 1, count), 0.2, 1.0)
    copy.hit_point = np.clip(copy.hit_point + nudge * 2.0 * rng.uniform(-1, 1, count), -1, 1)
    copy.facing = copy.facing + nudge * np.pi * rng.uniform(-1, 1, count)
    return copy


def make_world(count, rng, narrow=False):
    """A world of count objects (a multiple of 6), mixed like TABLE.

    Per six objects: three boxes, a roller, a spinner and a toy. Each object's numbers
    are drawn at random around TABLE's. A third of the objects arrive late: half of
    those in round 20, the other half in round 40.
    """
    kinds = ("box", "box", "box", "roller", "spinner", "toy") * (count // 6)
    reach_between = {"box": (0.3, 1.2), "roller": (1.0, 2.0), "spinner": (0.5, 1.0), "toy": (0.4, 0.8)}
    twist_between = {"box": (0.0, 0.0), "roller": (0.05, 0.3), "spinner": (1.0, 2.0), "toy": (0.5, 1.5)}
    reach = np.array([rng.uniform(*reach_between[kind]) for kind in kinds])
    twist = np.array([rng.uniform(*twist_between[kind]) for kind in kinds])

    late = rng.permutation(count)[:count // 3]
    appears_at = np.zeros(count, dtype=int)
    appears_at[late[:len(late) // 2]] = 20
    appears_at[late[len(late) // 2:]] = 40

    names = tuple(f"{kind} {number}" for number, kind in enumerate(kinds))
    return World(names, kinds, reach, twist, appears_at, narrow)


def narrowed(world):
    """The same world, but narrow."""
    return replace(world, narrow=True)


def _unit_vector(angle):
    """An arrow of length 1 pointing along each angle. One row per angle."""
    return np.stack([np.cos(angle), np.sin(angle)], axis=1)


TABLE = World(
    names=("wood", "steel", "rubber", "roller", "spinner", "toy"),
    kinds=("box", "box", "box", "roller", "spinner", "toy"),
    reach=np.array([1.0, 0.4, 0.7, 1.5, 0.8, 0.6]),
    twist=np.array([0.0, 0.0, 0.0, 0.1, 1.5, 1.0]),
    appears_at=np.array([0, 20, 0, 0, 40, 0]),
)

# Steps 1 and 2 use TABLE through these plain names.
OBJECTS = TABLE.names
TOY = OBJECTS.index("toy")
INPUT_SIZE = TABLE.input_size
random_pushes = TABLE.random_pushes
expected_result = TABLE.expected_result
push = TABLE.push
as_inputs = TABLE.as_inputs
