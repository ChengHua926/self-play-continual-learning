"""Montezuma's Revenge, the game the learners explore, through the Arcade Learning Environment.

The learner reads the game's 128 bytes of memory, never the screen. It can press one of 18
buttons: nothing, jump, the 8 directions, and the 8 directions with jump. Each press is held
for 4 frames, the usual frame skip. Sticky actions are off, so the game is deterministic: the
same memory and button always give the same next memory.

Every learner may jump back to a place it has saved, for free. The budget is frames played.

Two things here are only for measuring and reporting, never for choosing:
    room    memory byte 3 holds the room number
    lives   the emulator's count of lives left (memory byte 58 holds one less)

Places are told apart by screen_code: a tiny greyscale copy of the screen, 11 x 8 squares with
8 shades each. It knows nothing about the game.
"""

import numpy as np
from ale_py import ALEInterface, LoggerMode, roms

ALEInterface.setLoggerMode(LoggerMode.Error)

FRAME_SKIP = 4
MEMORY_SIZE = 128
BUTTONS = 18
ROOM_BYTE = 3


class Game:
    def __init__(self, seed):
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", seed)
        self.ale.setFloat("repeat_action_probability", 0.0)  # sticky actions off
        self.ale.setInt("frame_skip", FRAME_SKIP)
        self.ale.loadROM(roms.get_rom_path("montezuma_revenge"))
        self.buttons = self.ale.getMinimalActionSet()
        assert len(self.buttons) == BUTTONS
        self.frames_played = 0

    def press(self, button):
        """Hold one of the 18 buttons for FRAME_SKIP frames."""
        self.ale.act(self.buttons[button])
        self.frames_played += FRAME_SKIP

    def memory(self):
        return self.ale.getRAM().copy()

    def save(self):
        return self.ale.cloneState()

    def restore(self, saved):
        self.ale.restoreState(saved)

    def lives(self):
        return self.ale.lives()

    def over(self):
        return self.ale.game_over()

    def start_again(self):
        self.ale.reset_game()

    def screen_code(self):
        """The screen shrunk to 11 x 8 squares of 8 shades, as bytes, so that it can be a dictionary key."""
        screen = self.ale.getScreenGrayscale()[:208, :154].astype(float)  # 8 x 26 rows, 11 x 14 columns
        squares = screen.reshape(8, 26, 11, 14).mean(axis=(1, 3))
        return (squares // 32).astype(np.uint8).tobytes()


def change(before, after):
    """How each byte of memory moved, the short way round.

    A byte rolls over like an odometer: after 255 comes 0. So 255 -> 0 counts as +1, not -255,
    and a clock that ticks every step moves by +1 every step.
    """
    return (after.astype(np.int16) - before.astype(np.int16) + 128) % 256 - 128
