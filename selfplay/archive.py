"""The two stores of past programs that feed each round's pool (Cowsik et al., Appendix G).

The archive keeps the best few programs of each kind, as parents for mutation. A program's kind
(its niche) is its loop depth (machine.py), 0 to 8 with deeper loops counted as 8, and its body
length: up to 8 symbols, 9-16, 17-32, or more. That makes 9 x 4 = 36 niches. Only programs with
a positive reward get in, and each niche keeps its ELITES programs with the highest stored
reward. Stored rewards shrink by DECAY every round, so programs that were useful long ago give
way to newly useful ones. A parent is drawn by first picking an occupied niche uniformly, then
a program in it uniformly, so rare kinds (deep loops, say) are drawn as often as common ones.

The replay bank keeps every fresh and mutated program of earlier rounds. Each round draws
some uniformly, without repeats, and runs them again on new random bytes.
"""

from generator import MAX_PROGRAM
from model import END, INSTRUCTIONS, PROGRAM_TAG

DEEPEST = 8
LENGTH_EDGES = (8, 16, 32)
ELITES = 8
DECAY = 0.97


def body(program):
    """The program's symbols: tag S removed, cut at the first F."""
    return program[1:].split(bytes([END]), 1)[0]


def niche(program, depth):
    length = len(body(program))
    return min(depth, DEEPEST), sum(length > edge for edge in LENGTH_EDGES)


def mutate(program, rng):
    """The program with one symbol replaced, inserted or deleted, at a random place.

    New symbols come from the 18 instructions (never F). The result keeps at most MAX_PROGRAM - 1
    symbols, so that with its F it is no longer than a program the generator can write.
    """
    symbols = bytearray(body(program))
    edit = rng.integers(3) if symbols else 1                    # an empty program can only grow
    if edit == 1 and len(symbols) >= MAX_PROGRAM - 1:
        edit = 0                                                 # a full program cannot grow
    if len(symbols) == MAX_PROGRAM:
        edit = 2                                                 # a cut-off one (no F) must shrink to fit its F
    if edit == 0:                                                # replace, by a different symbol
        place = rng.integers(len(symbols))
        others = [s for s in INSTRUCTIONS if s != symbols[place]]
        symbols[place] = others[rng.integers(len(others))]
    elif edit == 1:                                              # insert
        symbols.insert(rng.integers(len(symbols) + 1), INSTRUCTIONS[rng.integers(len(INSTRUCTIONS))])
    else:                                                        # delete
        del symbols[rng.integers(len(symbols))]
    return bytes([PROGRAM_TAG]) + bytes(symbols) + bytes([END])


class Archive:
    def __init__(self):
        self.niches = {}               # niche -> {program: stored reward}

    def __len__(self):
        return sum(len(elites) for elites in self.niches.values())

    def decay(self):
        for elites in self.niches.values():
            for program in elites:
                elites[program] *= DECAY

    def admit(self, programs, depths, rewards):
        """Offer programs with this round's rewards. A program already in its niche gets its new reward."""
        for program, depth, reward in zip(programs, depths, rewards):
            if reward > 0:
                elites = self.niches.setdefault(niche(program, depth), {})
                elites[program] = reward
                if len(elites) > ELITES:
                    del elites[min(elites, key=elites.get)]

    def mutants(self, count, rng):
        """count mutated programs, each from a parent drawn as described at the top of this file.
        The archive must not be empty."""
        occupied = list(self.niches.values())
        children = []
        for _ in range(count):
            parents = list(occupied[rng.integers(len(occupied))])
            children.append(mutate(parents[rng.integers(len(parents))], rng))
        return children


class ReplayBank:
    def __init__(self):
        self.programs = []
        self.sampled_log_probs = []    # the generator's log-probability when it wrote the program; nan if mutated

    def __len__(self):
        return len(self.programs)

    def add(self, programs, sampled_log_probs):
        self.programs += programs
        self.sampled_log_probs += [float(x) for x in sampled_log_probs]

    def draw(self, count, rng):
        """Up to count programs, uniformly without repeats, with their sampling log-probabilities."""
        chosen = rng.choice(len(self), size=min(count, len(self)), replace=False) if len(self) else []
        return [self.programs[i] for i in chosen], [self.sampled_log_probs[i] for i in chosen]
