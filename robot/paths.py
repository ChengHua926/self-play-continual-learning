"""The robot runs reuse the paper's code in ../selfplay unchanged: the generator, its machine and
archive, the transformer blocks and the paper's reward. Importing this module makes them importable
(after this folder's own modules, which keep their names apart from selfplay's)."""

import sys
from pathlib import Path

SELFPLAY = str(Path(__file__).resolve().parent.parent / "selfplay")
if SELFPLAY not in sys.path:
    sys.path.append(SELFPLAY)
