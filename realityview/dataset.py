"""Reading the rendered NuRec dataset (see scripts/nurec/render_dataset.py for the format).

<scene>/meta.json: {rig: [...], frames: [{t_us, ego2world, views: [{file, kind: in|chase|orbit, name, f, c2rig, appearance}]}]}
<scene>/<file>.jpg  RGB (640x352);  <scene>/<file>_d.png  sparse lidar depth, uint16, metres * 256
"""
import glob
import json
import os
import cv2
import numpy as np


def list_scenes(data_dir, verbose=True):
    """-> [(scene_dir, meta)] for complete scenes. Scenes without rear cameras have NaN rig poses and are skipped."""
    out = []
    for mp in sorted(glob.glob(os.path.join(data_dir, "*", "meta.json"))):
        meta = json.load(open(mp))
        if all(np.isfinite(np.array(r["c2rig"])).all() for r in meta["rig"]):
            out.append((os.path.dirname(mp), meta))
        elif verbose:
            print("skip (missing rear cameras)", os.path.basename(os.path.dirname(mp)))
    return out


def split_views(frame):
    """-> (6 input views in rig order F, FL, FR, BL, BR, B; novel target views)."""
    return [v for v in frame["views"] if v["kind"] == "in"], [v for v in frame["views"] if v["kind"] != "in"]


def read_rgb(scene_dir, view):
    return cv2.imread(os.path.join(scene_dir, view["file"] + ".jpg"))[..., ::-1].copy()


def read_lidar_depth(scene_dir, view):
    return cv2.imread(os.path.join(scene_dir, view["file"] + "_d.png"), -1).astype(np.float32) / 256
