"""Render free-viewpoint teacher tracks for the video student, on top of the chase renders of render_teacher.py.

    python scripts/nurec/render_orbit.py [--teacher $RV_WORK/teacher] [--tracks 6]

The chase renders cover two fixed cameras behind the car; a student trained on them only works there. Each orbit track
is a rig-fixed camera that moves around the car the way a user drags it in the viewer: keyframes every ~1.5 s with
random yaw (full circle, spread over the tracks), pitch, distance and look-at point, smoothly interpolated, with some
holds. Per frame k and track i:
    orbit<i>/<k>.jpg, <k>_ego.png      appended to meta["frames"][k]["chase"] with its c2rig and view parameters
meta["chase"] gets one entry per track (kind "orbit"). The original meta.json is kept as meta_chase.json.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Scene, available_scenes, connect, look_at, write  # noqa: E402
from realityview import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default=os.path.join(paths.WORK, "teacher"))
ap.add_argument("--tracks", type=int, default=6)
ap.add_argument("--key_every", type=int, default=15, help="frames between keyframes")
ap.add_argument("--port", type=int, default=46435)
args = ap.parse_args()

# NuRec is reconstructed from car-height cameras: above ~40 deg pitch or ~5 m height the renders smear and the
# camera ends up inside tree canopies, so the range stays at what the teacher can render (the viewer clamps to it).
PITCH, DIST, AHEAD, MAX_HEIGHT = (8.0, 35.0), (6.0, 12.0), (0.0, 4.0), 5.0


def smooth(keys, n, every):
    """piecewise smoothstep through keyframes placed every `every` frames."""
    out = np.zeros((n, keys.shape[1]))
    for k in range(n):
        i, a = divmod(k, every)
        a = a / every
        a = a * a * (3 - 2 * a)
        out[k] = keys[i] * (1 - a) + keys[i + 1] * a
    return out


def track(rng, n, yaw0):
    """[n, 4] = yaw, pitch, dist, ahead per frame."""
    nk = n // args.key_every + 2
    keys = np.zeros((nk, 4))
    keys[0] = [yaw0, rng.uniform(*PITCH), rng.uniform(*DIST), rng.uniform(*AHEAD)]
    for i in range(1, nk):
        if rng.random() < 0.3:                 # hold: a static view for a while
            keys[i] = keys[i - 1]; continue
        keys[i] = [keys[i - 1, 0] + rng.uniform(-90, 90),
                   np.clip(keys[i - 1, 1] + rng.uniform(-30, 30), *PITCH),
                   np.clip(keys[i - 1, 2] + rng.uniform(-4, 4), *DIST),
                   rng.uniform(*AHEAD) if rng.random() < 0.3 else keys[i - 1, 3]]
    tr = smooth(keys, n, args.key_every)
    # cap the camera height (0.6 m look-at + dist * sin(pitch)) by lowering the pitch
    tr[:, 1] = np.minimum(tr[:, 1], np.degrees(np.arcsin(np.clip((MAX_HEIGHT - 0.6) / tr[:, 2], 0, 1))))
    return tr


def view_c2rig(yaw, pitch, dist, ahead):
    """same as realityview.geometry.orbit_camera (this script runs in the renderer client env, without torch)."""
    y, p = math.radians(yaw), math.radians(pitch)
    t = np.array([1.4 + ahead, 0.0, 0.6])
    return look_at(t + dist * np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)]), t)


def ego_mask(c2rig, box, f, w, h):
    p = (np.linalg.inv(c2rig) @ np.c_[box, np.ones(len(box))].T)[:3].T
    m = np.zeros((h, w), np.uint8)
    if (p[:, 2] > 0.1).all():
        uv = np.stack([p[:, 0] / p[:, 2] * f + w / 2, p[:, 1] / p[:, 2] * f + h / 2], 1)
        cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 255)
    return m


def main():
    st = connect(args.port)
    for scene_id in available_scenes(st):
        od = os.path.join(args.teacher, scene_id)
        mp = os.path.join(od, "meta.json")
        if not os.path.exists(mp):
            print("no chase teacher, skip", scene_id); continue
        meta = json.load(open(mp))
        if any(c.get("kind") == "orbit" for c in meta["chase"]):
            print("skip", scene_id); continue
        if not os.path.exists(os.path.join(od, "meta_chase.json")):
            shutil.copy(mp, os.path.join(od, "meta_chase.json"))
        sc = Scene(st, scene_id)
        rng = np.random.default_rng(zlib.crc32(scene_id.encode()))
        n, w, h, f = len(meta["frames"]), meta["chase_w"], meta["chase_h"], meta["chase_f"]
        box = np.array(meta["ego_box"])
        tracks = [track(rng, n, 180 + 360 * i / args.tracks + rng.uniform(-20, 20)) for i in range(args.tracks)]
        base_ci = len(meta["chase"])
        for i in range(args.tracks):
            meta["chase"].append(dict(kind="orbit", track=i, back=None, height=None))
        t0 = time.time()
        for k, fr in enumerate(meta["frames"]):
            E = np.array(fr["ego2world"])
            for i, tr in enumerate(tracks):
                yaw, pitch, dist, ahead = tr[k]
                c2rig = view_c2rig(yaw, pitch, dist, ahead).astype(np.float64)
                lid = sc.nearest_id(c2rig[:3, 2])
                fb = f"orbit{i}/{k:04d}"
                write(os.path.join(od, fb + ".jpg"), sc.render_rgb(fr["t_us"], E @ c2rig, f, w, h, lid, quality=95))
                write(os.path.join(od, fb + "_ego.png"), ego_mask(c2rig, box, f, w, h))
                fr["chase"].append(dict(file=fb, c2rig=c2rig.tolist(), appearance=lid, on_path=True,
                                        view=[float(yaw % 360), float(pitch), float(dist), float(ahead)]))
            if k % 25 == 0:
                print(f"{scene_id} {k}/{n}  {(k + 1) * args.tracks / (time.time() - t0):.1f} img/s", flush=True)
        json.dump(meta, open(mp + ".tmp", "w"))
        os.replace(mp + ".tmp", mp)
        print("done", scene_id, n * args.tracks, "images", f"{time.time() - t0:.0f}s", "chase index from", base_ci, flush=True)


if __name__ == "__main__":
    main()
