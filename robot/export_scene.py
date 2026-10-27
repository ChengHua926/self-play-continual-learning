"""Write the fixed world's scene out of robosuite, so that other MuJoCo versions can load it.

The scene file (about 50 kB) names about 70 mesh and texture files inside robosuite's package
(about 48 MB). This copies them next to it and points the scene at the copies, so the folder works
on its own, on any machine:
    data/robot_scene/square.xml
    data/robot_scene/assets/...
(data/ is not in git; copy the folder to another machine, or run this there.)

Run it with:   ../.venv-robosuite/bin/python export_scene.py
"""

import re
import shutil
from pathlib import Path

from world import World

FOLDER = Path(__file__).resolve().parent.parent / "data" / "robot_scene"


def main():
    xml = World().env.sim.model.get_xml()
    (FOLDER / "assets").mkdir(parents=True, exist_ok=True)
    copied = {}
    for original in sorted(set(re.findall(r'file="([^"]+)"', xml))):
        source = Path(original).resolve()
        name = f"{len(copied):02d}_{source.name}"         # numbered: different folders reuse file names
        shutil.copyfile(source, FOLDER / "assets" / name)
        copied[original] = f"assets/{name}"
    xml = re.sub(r'file="([^"]+)"', lambda match: f'file="{copied[match.group(1)]}"', xml)
    xml = re.sub(r'(meshdir|texturedir)="[^"]*"', r'\1="."', xml)
    (FOLDER / "square.xml").write_text(xml)
    print(f"wrote {FOLDER / 'square.xml'} and {len(copied)} asset files")


if __name__ == "__main__":
    main()
