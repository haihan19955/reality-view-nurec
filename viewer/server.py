"""Interactive Reality View viewer on a preprocessed scene (by default the first held-out validation scene).

    python viewer/server.py --ckpt work/checkpoints/step07.5k.pt [--scene 10] [--port 8770]

GET /              viewer page
GET /meta          {"n", "episode", "step"}
GET /surround?i=   jpeg mosaic of the 6 input cameras
GET /render?i=&yaw=&pitch=&dist=&geo=1&comp=1&car=1&steps=1&seed=0&what=gen|geo   -> jpeg (X-Ms header = server time)
"""
import argparse
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

from realityview import paths
from realityview.dataset import read_rgb, split_views
from realityview.geometry import orbit_camera
from realityview.pipeline import RealityView

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cache", default=paths.CACHE)
ap.add_argument("--scene", type=int, default=None, help="scene index in the cache (default: first validation scene)")
ap.add_argument("--port", type=int, default=8770)
ap.add_argument("--max_cached_frames", type=int, default=96)
args = ap.parse_args()

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
cache_meta = json.load(open(os.path.join(args.cache, "index.json")))
index = cache_meta["index"]
tokens = np.load(os.path.join(args.cache, "tok.npy"), mmap_mode="r")
rv = RealityView(args.ckpt)
SC = args.scene if args.scene is not None else rv.val_scenes[0]
SCENE_DIR = cache_meta["episodes"][SC]
rows = [k for k, (s, _) in enumerate(index) if s == SC]
scene_meta = json.load(open(os.path.join(SCENE_DIR, "meta.json")))
lock, frames = threading.Lock(), {}
print("scene", os.path.basename(SCENE_DIR), "frames", len(rows), "ckpt step", rv.step, flush=True)


def frame(i):
    if i in frames:
        return frames[i]
    ins, _ = split_views(scene_meta["frames"][index[rows[i]][1]])
    rgbs = [read_rgb(SCENE_DIR, v) for v in ins]
    state = rv.prepare(rgbs, [v["f"] for v in ins], [v["c2rig"] for v in ins], tokens[rows[i]])
    order = [1, 0, 2, 3, 5, 4]   # FL F FR / BL B BR
    mosaic = np.vstack([np.hstack([rgbs[o] for o in order[:3]]), np.hstack([rgbs[o] for o in order[3:]])])
    state["mosaic"] = cv2.imencode(".jpg", cv2.resize(mosaic[..., ::-1], (720, 264)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    if len(frames) >= args.max_cached_frames:
        frames.pop(next(iter(frames)))
    frames[i] = state
    return state


def render(q):
    state = frame(int(q.get("i", 0)) % len(rows))
    cam = orbit_camera(float(q.get("yaw", 180)), float(q.get("pitch", 20)), float(q.get("dist", 7)))
    img = rv.render(state, cam, geometry=q.get("geo", "1") == "1", composite=q.get("comp", "1") == "1", car=q.get("car", "1") == "1",
                    steps=int(q.get("steps", 1)), seed=int(q.get("seed", 0)), geometry_only=q.get("what") == "geo")
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype, extra=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        try:
            if u.path == "/":
                self.send(open(PAGE, "rb").read(), "text/html; charset=utf-8")
            elif u.path == "/meta":
                self.send(json.dumps({"n": len(rows), "episode": "NuRec " + os.path.basename(SCENE_DIR)[7:15], "step": rv.step}).encode(), "application/json")
            elif u.path == "/surround":
                with lock:
                    body = frame(int(q.get("i", 0)) % len(rows))["mosaic"]
                self.send(body, "image/jpeg")
            elif u.path == "/render":
                with lock:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    body = render(q)
                    torch.cuda.synchronize()
                    ms = (time.time() - t0) * 1000
                self.send(body, "image/jpeg", {"X-Ms": f"{ms:.1f}", "Access-Control-Expose-Headers": "X-Ms"})
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(repr(e).encode())
            print("ERR", repr(e), flush=True)


print(f"serving on http://localhost:{args.port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
