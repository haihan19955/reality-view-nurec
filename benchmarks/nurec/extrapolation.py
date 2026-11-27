"""How good are NuRec (3D Gaussian) renders away from the recorded trajectory?

There is no ground truth off the trajectory, so two proxies are measured against how far the
camera is moved from the recorded camera position:

  consistency : lidar points (rendered from a roof lidar) are coloured from the on-trajectory
                camera, re-projected into the moved camera and compared with its render there
                (PSNR on co-visible points; a view-consistent reconstruction keeps this high)
  sharpness   : mean gradient magnitude of the render relative to the on-trajectory render

Moves: lateral shift (rig y), lift (rig z, pitched down to keep the road in view), and the
chase / orbit cameras used as training targets. Needs a running renderer (start_renderer.sh).

    python benchmarks/nurec/extrapolation.py --scene <id> [--frames 8]
"""
import argparse
import json
import os
import sys

import cv2
import grpc
import numpy as np
from scipy.spatial.transform import Rotation as R

from realityview import paths

sys.path.insert(0, paths.NUREC_PROTO)
from nre.grpc.protos import common_pb2 as cm, sensorsim_pb2 as ss, sensorsim_pb2_grpc as ssg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="")
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--port", type=int, default=46435)
ap.add_argument("--out", default=os.path.join(paths.OUTPUTS, "nurec_extrapolation"))
args = ap.parse_args()
W, H, F = 640, 352, 320.0          # 90 deg hfov, same as the training targets
LATERAL = [0, 0.25, 0.5, 1, 2, 3, 5]
LIFT = [0, 0.5, 1, 2, 3, 5]
os.makedirs(args.out, exist_ok=True)

st = ssg.SensorsimServiceStub(grpc.insecure_channel(f"localhost:{args.port}", options=[("grpc.max_receive_message_length", 1 << 30)]))


def p2m(p):
    m = np.eye(4); m[:3, :3] = R.from_quat([p.quat.x, p.quat.y, p.quat.z, p.quat.w]).as_matrix(); m[:3, 3] = [p.vec.x, p.vec.y, p.vec.z]; return m


def m2p(m):
    q = R.from_matrix(m[:3, :3]).as_quat()
    return cm.Pose(vec=cm.Vec3(x=m[0, 3], y=m[1, 3], z=m[2, 3]), quat=cm.Quat(x=q[0], y=q[1], z=q[2], w=q[3]))


def cam_rot(f):
    f = f / np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); return np.stack([r, np.cross(f, r), f], 1)


def cam(pos, look):
    t = np.eye(4); t[:3, :3] = cam_rot(np.asarray(look, float) - pos); t[:3, 3] = pos; return t


scene = [s for s in st.get_available_scenes(cm.Empty()).scene_ids if args.scene in s][0]
cams = st.get_available_cameras(ss.AvailableCamerasRequest(scene_id=scene)).available_cameras
traj = st.get_available_trajectories(ss.AvailableTrajectoriesRequest(scene_id=scene)).available_trajectories[0].trajectory.poses
real = {c.logical_id: p2m(c.rig_to_camera) for c in cams}
front_id = max(real, key=lambda k: (real[k][:3, 2] @ [1, 0, 0]) - (0.5 if "tele" in k else 0))
mount = real[front_id][:3, 3]
shutter = cams[0].intrinsics.shutter_type
spec = ss.CameraSpec(logical_id=front_id, resolution_h=H, resolution_w=W, shutter_type=shutter)
spec.opencv_fisheye_param.CopyFrom(ss.OpenCVFisheyeCameraParam(principal_point_x=W / 2, principal_point_y=H / 2, focal_length_x=F, focal_length_y=F,
                                                                radial_coeffs=[1 / 3, 2 / 15, 17 / 315, 62 / 2835], max_angle=np.radians(80)))


def render(t_us, c2w):
    req = ss.RGBRenderRequest(scene_id=scene, resolution_h=H, resolution_w=W, camera_intrinsics=spec, frame_start_us=t_us, frame_end_us=t_us + 1,
                              sensor_pose=ss.PosePair(start_pose=m2p(c2w), end_pose=m2p(c2w)), image_format=ss.PNG)
    return cv2.imdecode(np.frombuffer(st.render_rgb(req).image_bytes, np.uint8), cv2.IMREAD_COLOR)


def project(pts_w, c2w):
    """pixel coords, depth, in-image & visible mask (z-buffer over a 5x5 neighbourhood of the sparse lidar)."""
    p = (np.linalg.inv(c2w) @ np.c_[pts_w, np.ones(len(pts_w))].T)[:3].T
    z = p[:, 2]; ok = z > 0.5
    u = np.where(ok, p[:, 0] / np.maximum(z, 1e-3) * F + W / 2, -1); v = np.where(ok, p[:, 1] / np.maximum(z, 1e-3) * F + H / 2, -1)
    ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    zb = np.full((H, W), 1e9, np.float32)
    ui, vi = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int)
    np.minimum.at(zb, (vi, ui), z[ok])
    zb = cv2.erode(zb, np.ones((5, 5), np.uint8))                       # local minimum depth
    vis = ok.copy(); vis[ok] = z[ok] < zb[vi, ui] * 1.03 + 0.05
    return u, v, z, vis


def sample(img, u, v):
    """bilinear colour at (u, v); out-of-image points get clamped values and are masked out by the caller."""
    u = np.clip(u, 0, W - 1.001); v = np.clip(v, 0, H - 1.001)
    x0, y0 = u.astype(int), v.astype(int); ax, ay = (u - x0)[:, None], (v - y0)[:, None]
    im = img.astype(np.float32)
    return (im[y0, x0] * (1 - ax) * (1 - ay) + im[y0, x0 + 1] * ax * (1 - ay)
            + im[y0 + 1, x0] * (1 - ax) * ay + im[y0 + 1, x0 + 1] * ax * ay)


def grad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))))


def psnr(a, b):
    return float(10 * np.log10(255 ** 2 / max(np.mean((a - b) ** 2), 1e-6)))


rng = np.random.default_rng(0)
idx = np.linspace(10, len(traj) - 10, args.frames).astype(int)
rows, sheet = [], None
for n, k in enumerate(idx):
    t_us = int(traj[k].timestamp_us); E = p2m(traj[k].pose)
    lid = np.eye(4); lid[:3, 3] = [0.8, 0, mount[2] + 0.3]
    lr = st.render_lidar(ss.LidarRenderRequest(scene_id=scene, lidar_config=ss.LidarSpec(lidar_type=ss.PANDAR128), frame_start_us=t_us, frame_end_us=t_us + 1,
                                               sensor_pose=ss.PosePair(start_pose=m2p(E @ lid), end_pose=m2p(E @ lid))))
    pts = np.array(lr.point_xyzs, np.float32).reshape(-1, 3)
    pts_rig = (lid @ np.c_[pts, np.ones(len(pts))].T)[:3].T
    pts_w = (E @ np.c_[pts_rig, np.ones(len(pts))].T)[:3].T
    look = mount + [10, 0, -mount[2]]                                 # road point 10 m ahead
    ref_T = cam(mount, mount + [10, 0, 0])
    ref = render(t_us, E @ ref_T)
    u, v, _, vis_ref = project(pts_w, E @ ref_T)
    col_ref = sample(ref, u, v)
    near = np.linalg.norm(pts_rig[:, :2], axis=1) < 12
    g_ref = grad(ref)

    moves = [("lateral", d, cam(mount + [0, d, 0], mount + [10, d, 0])) for d in LATERAL]
    moves += [("lift", d, cam(mount + [0, 0, d], look if d > 0 else mount + [10, 0, 0])) for d in LIFT]
    for j in range(4):                                                # training-target style views
        back, up, lat = rng.uniform(3, 8), rng.uniform(1, 3.5), rng.uniform(-3, 3)
        moves.append(("chase", float(np.hypot(back, np.hypot(up, lat))), cam(np.array([-back, lat, up]), [rng.uniform(4, 15), lat * rng.uniform(0, 1), 0.5])))
        a = rng.uniform(0, 2 * np.pi); rad = rng.uniform(5, 9); up = rng.uniform(2, 4.5)
        pos = np.array([1.4 + rad * np.cos(a), rad * np.sin(a), up])
        moves.append(("orbit", float(np.linalg.norm(pos - mount)), cam(pos, [1.4, 0, 0.6])))
    tiles = {}
    for kind, d, T in moves:
        img = ref if (kind in ("lateral", "lift") and d == 0) else render(t_us, E @ T)
        u2, v2, _, vis = project(pts_w, E @ T)
        m = vis & vis_ref
        c2 = sample(img, u2, v2)
        r = dict(frame=int(k), kind=kind, d=round(float(d), 2), n=int(m.sum()), sharp=grad(img) / g_ref,
                 psnr=psnr(c2[m], col_ref[m]) if m.sum() > 200 else None,
                 psnr_near=psnr(c2[m & near], col_ref[m & near]) if (m & near).sum() > 200 else None)
        rows.append(r)
        if n == 0 and d != 0.25:
            tiles.setdefault(kind, []).append((d, img))
    if n == 0:
        def strip(kind, cols):
            ts = [cv2.putText(cv2.resize(im, (320, 176)), f"{kind} {d:.1f} m", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2) for d, im in tiles[kind][:cols]]
            ts += [np.zeros((176, 320, 3), np.uint8)] * (cols - len(ts)); return np.hstack(ts)
        sheet = np.vstack([strip("lateral", 6), strip("lift", 6), np.hstack([strip("chase", 3), strip("orbit", 3)])])
        cv2.imwrite(os.path.join(args.out, f"{scene}_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"frame {n + 1}/{len(idx)}", flush=True)

json.dump(rows, open(os.path.join(args.out, f"{scene}.json"), "w"))
print("wrote", args.out, scene)
