"""Draw the CG ego car over a few dataset target views to check its pose and look.

    python tools/preview_ego_car.py [--data $NUREC_DATA]
"""
import argparse
import os

import cv2
import numpy as np

from realityview import ego_car, paths
from realityview.dataset import list_scenes, split_views
from realityview.geometry import TFX, TFY, TH, TW

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=paths.NUREC_DATA)
args = ap.parse_args()
d, meta = list_scenes(args.data)[0]
tiles = []
for fr in meta["frames"][10:60:10]:
    for v in split_views(fr)[1][:3]:
        img = cv2.resize(cv2.imread(os.path.join(d, v["file"] + ".jpg")), (TW, TH))
        tiles.append(ego_car.draw(img, np.array(v["c2rig"]), TFX, TFY))
tiles = tiles[:12]
os.makedirs(paths.OUTPUTS, exist_ok=True)
out = os.path.join(paths.OUTPUTS, "ego_car_preview.jpg")
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 12, 3)]))
print("wrote", out)
