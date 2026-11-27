"""Render continuous teacher sequences for the video student from a running NuRec renderer.

    python scripts/nurec/render_teacher.py [--out $RV_WORK/teacher] [--chase 6:2.5,9:4] [--max_frames N]

For every trajectory pose (10 Hz) of each scene:
  inputs/<k>_<cam>.jpg      the 6-camera input rig (640x352, same rig as render_dataset.py), + _d.png lidar depth
  chase<i>/<k>.jpg          a chase camera that follows the ego along its own past path (BACK metres behind
                            along the smoothed trajectory, HEIGHT metres up, looking LOOK_AHEAD metres ahead
                            of the ego). Staying on the driven path keeps NuRec close to where it was trained;
                            the main extrapolation is the height.
  chase<i>/<k>_d.png        sparse lidar depth (uint16, metres * 256)
  chase<i>/<k>_ego.png      ego-vehicle box mask (255 = ego); NuRec has no ego car and no supervised road under it
  meta.json                 poses (ego2world, c2rig per view), intrinsics, chase configs
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IN_H, IN_W, Scene, available_scenes, connect, look_at, project_depth, write  # noqa: E402
from realityview import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(paths.WORK, "teacher"))
ap.add_argument("--scenes", default="", help="comma-separated scene id substrings; default all")
ap.add_argument("--chase", default="6:2.5,9:4", help="BACK:HEIGHT metres, comma-separated")
ap.add_argument("--look_ahead", type=float, default=8.0)
ap.add_argument("--tw", type=int, default=832)
ap.add_argument("--th", type=int, default=480)
ap.add_argument("--tf", type=float, default=416.0, help="chase focal px (416 at 832 wide = 90 deg hfov)")
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--port", type=int, default=46435)
args = ap.parse_args()
CHASE = [tuple(float(x) for x in c.split(":")) for c in args.chase.split(",")]
EGO_BOX = np.array([[x, y, z] for x in (-1.1, 3.9) for y in (-1.0, 1.0) for z in (0.0, 1.7)])   # rig frame, metres


def path_point(s_query, pos, s, heading0):
    """point on the smoothed path at arc length s_query (extrapolated straight back before the start)."""
    if s_query <= 0:
        return pos[0] + heading0 * s_query
    i = min(np.searchsorted(s, s_query), len(s) - 1)
    a = (s_query - s[i - 1]) / max(s[i] - s[i - 1], 1e-6)
    return pos[i - 1] * (1 - a) + pos[i] * a


def ego_mask(c2w, ego2world, f, w, h):
    pts = (ego2world @ np.c_[EGO_BOX, np.ones(8)].T)[:3].T
    p = (np.linalg.inv(c2w) @ np.c_[pts, np.ones(8)].T)[:3].T
    m = np.zeros((h, w), np.uint8)
    if (p[:, 2] > 0.1).all():
        uv = np.stack([p[:, 0] / p[:, 2] * f + w / 2, p[:, 1] / p[:, 2] * f + h / 2], 1)
        cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 255)
    return m


st = connect(args.port)
scenes = available_scenes(st)
if args.scenes:
    scenes = [s for s in scenes if any(k in s for k in args.scenes.split(","))]
for scene_id in scenes:
    od = os.path.join(args.out, scene_id)
    if os.path.exists(os.path.join(od, "meta.json")):
        print("skip", scene_id); continue
    sc = Scene(st, scene_id)
    if not all(np.isfinite(r["T"]).all() for r in sc.rig):
        print("skip (missing rear cameras)", scene_id); continue
    n = len(sc.traj) if not args.max_frames else min(args.max_frames, len(sc.traj))
    E_all = [sc.pose(k)[1] for k in range(len(sc.traj))]
    pos = gaussian_filter1d(np.array([e[:3, 3] for e in E_all]), 3, axis=0, mode="nearest")
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    heading0 = E_all[0][:3, 0]
    meta = dict(scene=scene_id, hz=10, in_w=IN_W, in_h=IN_H, chase_w=args.tw, chase_h=args.th, chase_f=args.tf, depth_scale=256,
                ego_box=EGO_BOX.tolist(), look_ahead=args.look_ahead, chase=[dict(back=b, height=hh) for b, hh in CHASE],
                rig=[dict(name=r["name"], f=r["f"], cx=IN_W / 2, cy=IN_H / 2, c2rig=r["T"].tolist(), appearance=r["lid"]) for r in sc.rig],
                frames=[])
    t0 = time.time(); ground = []
    for k in range(n):
        t_us, E = sc.pose(k)
        pts_w = sc.lidar_world(t_us, E)
        if k % 20 == 0:   # ground height in the rig frame, for checking the ego box
            pr = (np.linalg.inv(E) @ np.c_[pts_w, np.ones(len(pts_w))].T)[:3].T
            near = pr[(np.abs(pr[:, 1]) < 2) & (pr[:, 0] > 5) & (pr[:, 0] < 12)]
            if len(near): ground.append(float(np.percentile(near[:, 2], 20)))
        fr = dict(k=k, t_us=t_us, ego2world=E.tolist(), chase=[])
        for r in sc.rig:
            c2w = E @ r["T"]; base = f"inputs/{k:04d}_{r['name']}"
            write(os.path.join(od, base + ".jpg"), sc.render_rgb(t_us, c2w, r["f"], IN_W, IN_H, r["lid"]))
            write(os.path.join(od, base + "_d.png"), project_depth(pts_w, c2w, r["f"], IN_W, IN_H))
        target = E @ np.array([args.look_ahead, 0, 0.6, 1])
        for i, (back, height) in enumerate(CHASE):
            cam_pos = path_point(s[k] - back, pos, s, heading0) + E[:3, 2] * height
            c2w = look_at(cam_pos, target[:3], up=E[:3, 2])
            c2rig = np.linalg.inv(E) @ c2w
            lid = sc.nearest_id(c2rig[:3, 2])
            base = f"chase{i}/{k:04d}"
            write(os.path.join(od, base + ".jpg"), sc.render_rgb(t_us, c2w, args.tf, args.tw, args.th, lid, quality=95))
            write(os.path.join(od, base + "_d.png"), project_depth(pts_w, c2w, args.tf, args.tw, args.th))
            write(os.path.join(od, base + "_ego.png"), ego_mask(c2w, E, args.tf, args.tw, args.th))
            # on_path=False: the camera is before the start of the clip (extrapolated), where NuRec is poor
            fr["chase"].append(dict(file=base, c2rig=c2rig.tolist(), appearance=lid, on_path=bool(s[k] - back >= 0)))
        meta["frames"].append(fr)
        if k % 25 == 0:
            print(f"{scene_id} {k}/{n}  {(k + 1) / (time.time() - t0):.2f} frames/s", flush=True)
    meta["ground_z_rig"] = float(np.median(ground)) if ground else None
    json.dump(meta, open(os.path.join(od, "meta.json"), "w"))
    print("done", scene_id, n, "frames", f"{time.time() - t0:.0f}s", "ground z (rig) %.2f" % (meta["ground_z_rig"] or float("nan")), flush=True)
