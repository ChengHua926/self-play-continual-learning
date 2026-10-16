"""The step-1 world: six objects on a table, and one thing you can do to them: push.

Nothing here is real physics. The rules are simple on purpose. What matters for step 1
is that the world has easy rules, harder rules, and one object that moves by pure chance.

    wood, steel, rubber   slide straight along the push. Same rule, different distance.
    roller                rolls far the way it is facing, and hardly at all sideways.
    spinner               hit it off-centre and it turns instead of sliding.
    toy                   ignores the push and moves at random. This is the noise.
"""

from dataclasses import dataclass

import numpy as np

OBJECTS = ("wood", "steel", "rubber", "roller", "spinner", "toy")
TOY = OBJECTS.index("toy")

# How far each box slides for a full-strength push.
SLIDE_DISTANCE = {"wood": 1.0, "steel": 0.4, "rubber": 0.7}

# The network sees 12 numbers per push and predicts 3.
INPUT_SIZE = len(OBJECTS) + 6
RESULT_SIZE = 3


@dataclass
class Pushes:
    """A batch of pushes. Every field is an array with one entry per push."""

    obj: np.ndarray        # which object, as an index into OBJECTS
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


def random_pushes(count, shares, rng):
    """Draw pushes at random.

    shares[i] says how often object i gets picked. A share of 0 means that object
    is not on the table.
    """
    shares = np.asarray(shares, dtype=float)
    return Pushes(
        obj=rng.choice(len(OBJECTS), size=count, p=shares / shares.sum()),
        direction=rng.uniform(0, 2 * np.pi, count),
        strength=rng.uniform(0.2, 1.0, count),
        hit_point=rng.uniform(-1, 1, count),
        facing=rng.uniform(0, 2 * np.pi, count),
    )


def expected_result(pushes):
    """What the rules say each push does: (moved along x, moved along y, turned).

    For the toy this is all zeros. The toy moves by chance, so zero is its average
    result, and the average is the best guess anyone can make.
    """
    result = np.zeros((len(pushes), RESULT_SIZE))
    moved = result[:, :2]   # writing into these two also writes into result
    turned = result[:, 2]

    power = (pushes.strength**2)[:, None]
    along_push = _unit_vector(pushes.direction)
    along_facing = _unit_vector(pushes.facing)
    across_facing = _unit_vector(pushes.facing + np.pi / 2)

    for name, distance in SLIDE_DISTANCE.items():
        box = pushes.obj == OBJECTS.index(name)
        moved[box] = (distance * power * along_push)[box]

    roller = pushes.obj == OBJECTS.index("roller")
    angle = (pushes.direction - pushes.facing)[:, None]
    rolled = 1.5 * power * np.cos(angle) * along_facing
    scraped = 0.1 * power * np.sin(angle) * across_facing
    moved[roller] = (rolled + scraped)[roller]

    spinner = pushes.obj == OBJECTS.index("spinner")
    off_centre = np.abs(pushes.hit_point)[:, None]
    moved[spinner] = (0.8 * power * (1 - off_centre) * along_push)[spinner]
    turned[spinner] = (1.5 * pushes.strength * pushes.hit_point)[spinner]

    return result


def push(pushes, rng):
    """Do the pushes and return what happened.

    Every object follows its rule exactly, except the toy, which ignores the push.
    """
    result = expected_result(pushes)
    toy = pushes.obj == TOY
    count = toy.sum()
    distance = rng.uniform(0, 0.6, count)[:, None]
    result[toy, :2] = distance * _unit_vector(rng.uniform(0, 2 * np.pi, count))
    result[toy, 2] = rng.uniform(-1, 1, count)
    return result


def as_inputs(pushes):
    """Turn pushes into the 12 numbers per push that the network reads.

    Six on/off switches say which object it is. Angles go in as (cos, sin) so that
    359 degrees and 1 degree look close to each other.
    """
    inputs = np.zeros((len(pushes), INPUT_SIZE))
    inputs[np.arange(len(pushes)), pushes.obj] = 1
    inputs[:, 6:8] = _unit_vector(pushes.direction)
    inputs[:, 8] = pushes.strength
    inputs[:, 9] = pushes.hit_point
    inputs[:, 10:12] = _unit_vector(pushes.facing)
    return inputs


def _unit_vector(angle):
    """An arrow of length 1 pointing along each angle. One row per angle."""
    return np.stack([np.cos(angle), np.sin(angle)], axis=1)
