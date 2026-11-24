"""List rendered scenes, their frame counts and any cameras with missing (NaN) mounts.

    python scripts/nurec/check_dataset.py [--data $NUREC_DATA]
"""
import argparse
import glob
import json
import os

import numpy as np

from realityview import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=paths.NUREC_DATA)
args = ap.parse_args()
bad = []
for mp in sorted(glob.glob(os.path.join(args.data, "*", "meta.json"))):
    m = json.load(open(mp))
    nan = [r["name"] for r in m["rig"] if not np.isfinite(np.array(r["c2rig"])).all()]
    name = os.path.basename(os.path.dirname(mp))
    print(name[:15], len(m["frames"]), "frames", "NaN cams:", nan, "appearance:", sorted({r["appearance"][7:] for r in m["rig"]}))
    if nan:
        bad.append(name)
print("scenes without rear cameras (skipped in training):", bad)
