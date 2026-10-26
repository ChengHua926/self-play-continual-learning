"""The program machine: Brainfuck with ten macro instructions, as in Cowsik et al. (Appendix E).

A program is the generator's bytes after the tag byte S, cut at the first F. Each macro byte is
replaced by its fixed Brainfuck body (MACROS), and the result runs on a circular tape of byte
cells:
    >  <     move the head right / left, wrapping around the tape
    +  -     add / subtract 1, modulo 256
    .        append the current cell to the output
    ,        write a random byte into the current cell
    [  ]     loop while the current cell is not 0; a bracket with no partner does nothing
Every executed instruction costs one step. A run stops at the end of the program, after
max_steps steps, or when the output holds OUTPUT_LENGTH bytes. The learner trains on the tag
byte O followed by the output, which is 0 after the last written byte.

A run also reports its loop depth: the most loops it was inside at once while running, counted on
the code after the macros are expanded (so W, whose body holds [-], counts as two). A loop
counts only if it is entered: a [ that finds 0 and skips its loop adds nothing. The archive
(archive.py) sorts programs by this depth.

The limits grow with the model, as theirs do (limits()). For the smallest model (width 64,
1 layer): 65,536 steps and 1,024 tape cells.

run() runs many programs at once on all CPU cores (numba). reference_run() is the same machine
written plainly, one program at a time, for the tests. Both draw the random bytes for , from
the same small generator (splitmix64), seeded per program, so they give identical outputs.
"""

import numba
import numpy as np

MACROS = {
    "Z": "[-]",               # clear the cell
    "R": "[->+<]",            # add the cell into the next one
    "L": "[->+++<]",          # add three times the cell into the next one
    "N": "[-<->]",            # subtract the cell from the previous one
    "C": "[->+>+<<]",         # copy the cell into the next two
    "G": "[>]",               # move right to a zero cell
    "H": "[<]",               # move left to a zero cell
    "W": "[[-]>+<]",          # next cell += (cell != 0); cell = 0
    "V": "[.>]",              # print cells until a zero cell
    "X": "[-]" + "+" * 16,    # set the cell to 16
}
OUTPUT_LENGTH = 4095
OUTPUT_TAG = ord("O")
_EXPANSION = {ord(k): v for k, v in MACROS.items()}


def limits(width, layers):
    """(max_steps, tape_length) for a model of this shape, as in training."""
    return max(2 ** 14, layers * width ** 2 * 16), max(1024, width * 8)


def expand(program):
    """The Brainfuck code of a program: tag S removed, cut at the first F, macros expanded."""
    body = bytes(program[1:]).decode("ascii", errors="ignore").split("F", 1)[0]
    return body.translate(_EXPANSION).encode("ascii")


def jumps_for(code):
    """For each position, where a bracket jumps to; a bracket with no partner jumps to itself."""
    jumps = np.arange(len(code), dtype=np.int64)
    open_brackets = []
    for i, c in enumerate(code):
        if c == ord("["):
            open_brackets.append(i)
        elif c == ord("]") and open_brackets:
            j = open_brackets.pop()
            jumps[i], jumps[j] = j, i
    return jumps


@numba.njit(inline="always")
def _splitmix64(state):
    state = (state + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = state
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return state, z ^ (z >> np.uint64(31))


@numba.njit(parallel=True, cache=True)
def _run_all(codes, jumps, starts, seeds, max_steps, tape_length, outputs, written, depths):
    for p in numba.prange(len(starts) - 1):
        start, end = starts[p], starts[p + 1]
        tape = np.zeros(tape_length, dtype=np.uint8)
        state = seeds[p]
        head = 0
        pc = start
        count = 0
        steps = 0
        depth = 0
        deepest = 0
        while steps < max_steps and pc < end and count < OUTPUT_LENGTH:
            c = codes[pc]
            if c == 62:                       # >
                head = (head + 1) % tape_length
            elif c == 60:                     # <
                head = (head - 1) % tape_length
            elif c == 43:                     # +
                tape[head] = (tape[head] + 1) % 256
            elif c == 45:                     # -
                tape[head] = (tape[head] + 255) % 256
            elif c == 46:                     # .
                outputs[p, 1 + count] = tape[head]
                count += 1
            elif c == 91:                     # [
                if tape[head] == 0:
                    pc = start + jumps[pc]
                elif jumps[pc] != pc - start:  # entering a loop (an unpartnered [ does nothing)
                    depth += 1
                    deepest = max(deepest, depth)
            elif c == 93:                     # ]
                if tape[head] != 0:
                    pc = start + jumps[pc]
                elif jumps[pc] != pc - start:  # leaving a loop
                    depth -= 1
            elif c == 44:                     # ,
                state, value = _splitmix64(state)
                tape[head] = value % np.uint64(256)
            pc += 1
            steps += 1
        written[p] = count
        depths[p] = deepest


def run(programs, seeds, max_steps, tape_length):
    """Run many programs. Returns three arrays, one row or entry per program:
        outputs  [program, 1 + OUTPUT_LENGTH] uint8: tag O, then the output (0 after the last byte)
        written  [program]: how many bytes the program wrote
        depths   [program]: its loop depth (see the top of this file)

    programs: a list of byte strings, each starting with the tag S. seeds: one integer per program,
    for its random bytes.
    """
    codes = [expand(p) for p in programs]
    starts = np.cumsum([0] + [len(c) for c in codes]).astype(np.int64)
    flat = np.frombuffer(b"".join(codes), dtype=np.uint8) if starts[-1] else np.zeros(1, np.uint8)
    jumps = np.concatenate([jumps_for(c) for c in codes]) if starts[-1] else np.zeros(1, np.int64)
    outputs = np.zeros((len(programs), 1 + OUTPUT_LENGTH), dtype=np.uint8)
    outputs[:, 0] = OUTPUT_TAG
    written = np.zeros(len(programs), dtype=np.int64)
    depths = np.zeros(len(programs), dtype=np.int64)
    _run_all(flat, jumps, starts, np.asarray(seeds, dtype=np.uint64), max_steps, tape_length,
             outputs, written, depths)
    return outputs, written, depths


def reference_run(program, seed, max_steps, tape_length):
    """The same machine, plainly, for one program; for the tests. Returns (output, written, depth)."""
    code, mask = expand(program), 0xFFFFFFFFFFFFFFFF
    jumps = jumps_for(code)
    tape, out = [0] * tape_length, [OUTPUT_TAG]
    state, head, pc, steps, depth, deepest = seed, 0, 0, 0, 0, 0
    while steps < max_steps and pc < len(code) and len(out) - 1 < OUTPUT_LENGTH:
        c = chr(code[pc])
        if c == ">":
            head = (head + 1) % tape_length
        elif c == "<":
            head = (head - 1) % tape_length
        elif c == "+":
            tape[head] = (tape[head] + 1) % 256
        elif c == "-":
            tape[head] = (tape[head] - 1) % 256
        elif c == ".":
            out.append(tape[head])
        elif c == "[" and tape[head] == 0:
            pc = jumps[pc]
        elif c == "[" and jumps[pc] != pc:
            depth += 1
            deepest = max(deepest, depth)
        elif c == "]" and tape[head] != 0:
            pc = jumps[pc]
        elif c == "]" and jumps[pc] != pc:
            depth -= 1
        elif c == ",":
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            tape[head] = (z ^ (z >> 31)) % 256
        pc += 1
        steps += 1
    written = len(out) - 1
    return np.array(out + [0] * (1 + OUTPUT_LENGTH - len(out)), dtype=np.uint8), written, deepest
