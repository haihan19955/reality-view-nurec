"""Live Reality View: drives through a NuRec scene and generates a view around the car in real time with the distilled
video student (MeanFlow, 1 step), conditioned on the 6 rig cameras projected into the current virtual camera.
The camera can be changed while it plays; the next 8-frame chunk is generated from the new viewpoint.

    python viewer/live.py [--scene e8cf41e8] [--ckpt $RV_WORK/checkpoints/student_mf_free/last.pt] [--port 8771]

GET /            page (viewer/live.html)
GET /next?after= next generated frame after sequence number `after` (jpeg; headers X-Seq, X-Frame, X-Cam, X-Stats)
GET /preview?frame=&yaw=&pitch=&dist=&ahead=   instant reprojection of generated frame `frame` into another view
                 (shown while dragging, until generated frames from the new view arrive)
GET /surround?i= the 6 input cameras of frame i (jpeg)
GET /set?yaw=&pitch=&dist=&ahead=&car=&inset=&pause=&restart=   change the view (orbit around the car) / playback

With a free-view checkpoint (train.py --free_view) the camera glides toward the requested view inside the generated
chunks and the previous frame stays the context (for big jumps it moves there directly with a fresh context). The
generator answers a view change only after ~1 s (8-frame chunks), so while the user drags the page shows /preview:
the last generated frame, lifted to 3D with its point-cloud depth and re-rendered in the new view (~10 ms), the
same idea as reprojection in VR headsets. A chase-only checkpoint restarts the context on every view change.
"""
import argparse
import collections
import glob
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

from realityview import ego_car, paths
from realityview.geometry import orbit_camera
from realityview.student import CAM_FRAMES, EGO_DIM, EGO_DIM_CAM, FEAT_TIMES, Student, ego_vector
from realityview.warp import PAST, Splatter, fill_holes, frame_depth, pack_mask, read_inputs

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="e8cf41e8")
ap.add_argument("--ckpt", default=os.path.join(paths.CHECKPOINTS, "student_mf_free", "last.pt"),
                help="distilled student (distill.py); a chase-only one (student_mf) also works, without free view")
ap.add_argument("--teacher", default=os.path.join(paths.WORK, "teacher"))
ap.add_argument("--cache", default=os.path.join(paths.WORK, "student_cache"))
ap.add_argument("--wan", default=paths.model("wan2.1-1.3b"))
ap.add_argument("--da3", default=paths.model("DA3METRIC-LARGE"))
ap.add_argument("--steps", type=int, default=1)
# tae = TAEHV (madebyollin/taehv, taew2_1.pth in $RV_MODELS/taehv), a tiny autoencoder distilled to Wan 2.1's
# normalized latent space: ~20x faster than the full Wan VAE, ~2 dB lower reconstruction PSNR
ap.add_argument("--enc", default="tae", choices=["tae", "full"], help="VAE that encodes the warp / context frame")
ap.add_argument("--dec", default="tae", choices=["tae", "full"], help="VAE that decodes the output")
ap.add_argument("--no_lidar_scale", action="store_true", help="DA3 metric depth alone (camera-only car)")
ap.add_argument("--feat", default="", help="camera feature cache (dino / dino3); default: from the checkpoint folder's feat.txt, else dino")
ap.add_argument("--port", type=int, default=8771)
args = ap.parse_args()
dev = "cuda"
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live.html")
VAL_SCENES = ("e8cf41e8", "ff28f961")


# orbit camera around (1.4 + ahead, 0, 0.6) in the rig frame; the range is what the NuRec teacher can render
# (scripts/nurec/render_orbit.py), the per-frame speed limit is about the fastest camera motion seen in training
LIMITS = dict(pitch=(5.0, 38.0), dist=(5.0, 13.0), ahead=(0.0, 5.0))
MAX_HEIGHT = 5.5
SPEED = dict(yaw=9.0, pitch=4.0, dist=0.6, ahead=0.5)          # per frame (10 Hz)


def clamp_view(p):
    p = dict(p)
    for k, (lo, hi) in LIMITS.items():
        p[k] = min(hi, max(lo, p[k]))
    p["pitch"] = min(p["pitch"], np.degrees(np.arcsin(min(1.0, (MAX_HEIGHT - 0.6) / p["dist"]))))
    return p


def view_c2rig(p):
    return orbit_camera(p["yaw"], p["pitch"], p["dist"], target=(1.4 + p["ahead"], 0.0, 0.6))


def step_toward(cur, tgt):
    """move the camera parameters one frame toward the requested view (shortest way round for yaw)."""
    out = {}
    for k, s in SPEED.items():
        d = tgt[k] - cur[k]
        if k == "yaw":
            d = (d + 180) % 360 - 180
        out[k] = cur[k] + max(-s, min(s, d))
    return out


JUMP_GAP = 30.0       # view_gap above which the generator jumps to the requested view instead of gliding


def view_gap(a, b):
    """rough distance between two orbit views in degrees (yaw / pitch) with 10 deg per metre of distance / ahead."""
    dy = abs((a["yaw"] - b["yaw"] + 180) % 360 - 180)
    return max(dy, abs(a["pitch"] - b["pitch"]), 10 * abs(a["dist"] - b["dist"]), 10 * abs(a["ahead"] - b["ahead"]))


class View:
    """Camera / playback state shared between the HTTP handlers and the generator thread."""

    def __init__(self):
        self.p = dict(yaw=180.0, pitch=16.0, dist=8.0, ahead=2.0)      # behind the car, chase-like
        self.car, self.inset, self.pause, self.restart = True, True, False, False
        self.version = 0
        self.token = "0"          # client's id of the view it asked for, echoed on every frame
        self.lock = threading.Lock()

    def set(self, q):
        with self.lock:
            old = dict(self.p)
            for k in self.p:
                if k in q:
                    self.p[k] = float(q[k])
            self.p = clamp_view(self.p)
            for k in ("car", "inset", "pause"):
                if k in q:
                    setattr(self, k, q[k] == "1")
            if "v" in q:
                self.token = q["v"]
            if q.get("restart") == "1":
                self.restart = True
            if old != self.p or self.restart:
                self.version += 1

    def cam(self):
        with self.lock:
            return dict(self.p), self.version, self.token


class Engine:
    def __init__(self):
        from diffusers import AutoencoderKLWan
        from realityview.depth import MetricDepth
        self.tdir = [d for d in glob.glob(os.path.join(args.teacher, "*")) if args.scene in d][0]
        self.cdir = [d for d in glob.glob(os.path.join(args.cache, "*")) if args.scene in d][0]
        self.meta = json.load(open(os.path.join(self.tdir, "meta.json")))
        ft = os.path.join(os.path.dirname(args.ckpt), "feat.txt")
        feat = args.feat or (open(ft).read().strip() if os.path.exists(ft) else "dino")
        self.dino = np.load(os.path.join(self.cdir, feat + ".npy"), mmap_mode="r")
        self.ego = np.load(os.path.join(self.cdir, "ego.npy"))
        self.N = len(self.meta["frames"])
        t0 = time.time()
        da3 = MetricDepth(args.da3)
        self.rgb, self.depth = [], []
        for fr in self.meta["frames"]:
            rgbs = read_inputs(self.tdir, fr["k"])
            self.rgb.append(torch.from_numpy(rgbs).to(dev))
            self.depth.append(torch.from_numpy(frame_depth(da3, rgbs, self.tdir, fr["k"], self.meta, not args.no_lidar_scale)).half().to(dev))
        del da3; torch.cuda.empty_cache()
        print(f"scene {self.meta['scene']}: {self.N} frames, depth {time.time() - t0:.0f}s", flush=True)
        self.E = [torch.tensor(f["ego2world"], device=dev, dtype=torch.float32) for f in self.meta["frames"]]
        self.sp = Splatter(self.meta, dev)
        self.pts_cache, self.pts_lock = {}, threading.RLock()
        self.gen = {}                  # frame -> (generated RGB HxWx3 uint8, its point-cloud depth, c2rig): for /preview
        sd = torch.load(args.ckpt, map_location=dev)
        # free-view checkpoints take the camera pose of frames CAM_FRAMES in the ego tokens; chase-only ones a fixed config
        self.free = sd["ego.0.weight"].shape[1] == EGO_DIM_CAM
        self.model = Student(args.wan, geo=True, meanflow=True, ego_dim=EGO_DIM_CAM if self.free else EGO_DIM).to(dev).eval()
        r = self.model.load_state_dict(sd, strict=False)
        print("model", args.ckpt, "free view" if self.free else "chase only", "features", feat, flush=True)
        assert not r.unexpected_keys, r.unexpected_keys
        if "full" in (args.enc, args.dec):
            self.vae = AutoencoderKLWan.from_pretrained(args.wan, subfolder="vae", torch_dtype=torch.bfloat16).to(dev).eval()
            self.lm = torch.tensor(self.vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
            self.ls = torch.tensor(self.vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
        if "tae" in (args.enc, args.dec):
            import sys
            sys.path.insert(0, paths.model("taehv"))
            from taehv import TAEHV
            self.tae = TAEHV(paths.model("taehv", "taew2_1.pth")).to(dev).half().eval()
        self.g = torch.Generator(device=dev).manual_seed(0)

    def enc(self, x):
        """[B,3,T,H,W] in [-1,1] -> normalized Wan latent [B,16,t,h,w]"""
        if args.enc == "full":
            return (self.vae.encode(x.bfloat16()).latent_dist.mode().float() - self.lm) / self.ls
        return self.tae.encode_video(((x + 1) / 2).transpose(1, 2).half(), show_progress_bar=False).transpose(1, 2).float()

    def dec(self, z):
        if args.dec == "full":
            return self.vae.decode((z * self.ls + self.lm).bfloat16()).sample.float().clamp(-1, 1)
        return self.tae.decode_video(z.transpose(1, 2).half(), show_progress_bar=False).transpose(1, 2).float() * 2 - 1

    def points(self, j):
        with self.pts_lock:            # shared by the generator thread and the /preview handlers
            if j not in self.pts_cache:
                self.pts_cache[j] = self.sp.points(self.depth[j].float(), self.rgb[j], self.E[j])
                for old in [x for x in self.pts_cache if abs(x - j) > 48]:   # generator window + what is on screen
                    del self.pts_cache[old]
            return self.pts_cache[j]

    def layers(self, j):
        return [self.points(j)] + [self.points(j - p) for p in PAST if j - p >= 0]

    def warp(self, j, c2rig):
        """point-cloud warp into the camera c2rig at frame j: RGB, observed mask, depth (all HxW tensors)."""
        return self.sp.render_fast(self.layers(j), self.E[j] @ c2rig, with_depth=True)

    @torch.no_grad()
    def reproject(self, j, c2rig_new):
        """Instant view while dragging (like VR reprojection): the last generated frame j, lifted to 3D with its
        point-cloud depth, rendered into the new camera; what it does not cover comes from the raw point cloud."""
        g = self.gen.get(j)
        if g is None:
            return None
        img, depth, c2rig = g
        gen = self.sp.unproject_view(img, depth, self.E[j] @ c2rig)
        out, m = self.sp.render_fast([gen] + self.layers(j), self.E[j] @ torch.as_tensor(c2rig_new, device=dev, dtype=torch.float32))
        return fill_holes(out, m).cpu().numpy()[..., ::-1]           # newly exposed areas / splat gaps

    @torch.no_grad()
    def chunk(self, k, c2rigs, zc):
        """8 new frames k+1..k+8 seen from c2rigs[1..8] (c2rigs[0] = camera of the context frame k)
        -> (list of BGR uint8, list of warp BGR uint8, latent of the last frame, timings)."""
        tm = {}
        t0 = time.time()
        ws, ms, ds = zip(*[self.warp(k + j, torch.tensor(c2rigs[j], device=dev, dtype=torch.float32)) for j in range(9)])
        torch.cuda.synchronize(); tm["warp"] = time.time() - t0
        x = torch.stack(ws).permute(3, 0, 1, 2)[None].float() / 127.5 - 1
        geo = torch.cat([pack_mask(torch.stack(ms)).to(dev), self.enc(x)[0]])[None]
        f = torch.from_numpy(np.stack([self.dino[k + o] for o in FEAT_TIMES]).astype(np.float32))[None].to(dev)
        if self.free:
            e = ego_vector(self.ego, k, cams=[c2rigs[o] for o in CAM_FRAMES])
        else:
            e = ego_vector(self.ego, k, dict(back=6.0, height=2.5))
        e = torch.from_numpy(e)[None].to(dev)
        torch.cuda.synchronize(); tm["encode"] = time.time() - t0 - tm["warp"]
        z = torch.randn(1, 16, 3, 60, 104, device=dev, generator=self.g)
        tc = 1.0 if zc is None else 0.1
        ctx = z[:, :, :1].clone() if zc is None else (1 - tc) * zc + tc * torch.randn(zc.shape, device=dev, generator=self.g)
        sig = torch.linspace(1, 0, args.steps + 1)
        sig = 3 * sig / (1 + 2 * sig)
        for i in range(args.steps):
            a, b = float(sig[i]), float(sig[i + 1])
            z[:, :, :1] = ctx
            u = self.model(z, torch.tensor([[tc, a, a]], device=dev), f, e, geo, torch.tensor([[0.0, a - b, a - b]], device=dev))
            z[:, :, 1:] = z[:, :, 1:] - (a - b) * u[:, :, 1:]
        torch.cuda.synchronize(); tm["generate"] = time.time() - t0 - tm["warp"] - tm["encode"]
        y = self.dec(z)[0, :, 1:]                                                     # 3, 8, H, W
        zc = self.enc(y[:, -1:][None])
        rgb = ((y + 1) * 127.5).round().byte().permute(1, 2, 3, 0)                   # 8, H, W, 3 on the GPU
        for j in range(8):
            self.gen[k + 1 + j] = (rgb[j], ds[j + 1], torch.as_tensor(c2rigs[j + 1], device=dev, dtype=torch.float32))
        for old in [x for x in self.gen if not k - 40 <= x <= k + 8]:
            del self.gen[old]
        frames = rgb.cpu().numpy()[..., ::-1]
        warps = [w.cpu().numpy()[..., ::-1] for w in ws[1:]]
        torch.cuda.synchronize(); tm["decode"] = time.time() - t0 - sum(tm.values())
        tm["total"] = time.time() - t0
        return list(frames), warps, zc, tm


def compose(img, warp, c2rig, f, car, inset):
    out = ego_car.draw(np.ascontiguousarray(img), c2rig, f, f) if car else img.copy()
    if inset:
        h, w = out.shape[:2]
        s = cv2.resize(np.ascontiguousarray(warp), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        out[8:8 + s.shape[0], w - 8 - s.shape[1]:w - 8] = s
        cv2.rectangle(out, (w - 9 - s.shape[1], 7), (w - 8, 8 + s.shape[0]), (255, 255, 255), 1)
    return out


VIEW = View()
OUT = collections.deque()           # (seq, frame index, jpeg)
OUT_COND = threading.Condition()
STATS = {"status": "loading", "fps": 0.0}
SEQ = [0]


def generator():
    eng = Engine()
    GLOBAL["eng"] = eng
    STATS["scene"] = eng.meta["scene"][7:15]
    STATS["held_out"] = any(v in eng.meta["scene"] for v in VAL_SCENES)
    STATS["n"] = eng.N
    STATS["free"] = eng.free
    k0 = max(PAST)
    k, zc, ver = k0, None, -1
    cur = None                           # camera parameters of the last generated frame
    ema = None
    # free-view model: the camera glides toward the requested view inside the chunks and the previous frame stays the
    # context (no restart), so dragging is continuous; only a little playback is buffered so the camera follows fast
    ahead_frames = 2 if eng.free else 12
    while True:
        if VIEW.pause:
            STATS["status"] = "paused"; time.sleep(0.05); continue
        tgt, v, token = VIEW.cam()
        if v != ver:
            ver = v
            if VIEW.restart:
                VIEW.restart, k, zc, cur = False, k0, None, None
            if not eng.free:             # chase-only model: the previous frame is in another view, start a fresh context
                zc, cur = None, None
                with OUT_COND:
                    OUT.clear()
        if k + 8 >= eng.N:
            k, zc = k0, None
        if cur is None or not eng.free:
            cur = dict(tgt)
        elif view_gap(cur, tgt) > JUMP_GAP:
            # far from the requested view: the client shows the instant reprojection meanwhile, so jump straight there
            # and start a fresh context (gliding 180 deg at the trained camera speed would take ~2 s)
            cur, zc = dict(tgt), None
        ps = [cur]
        for _ in range(8):
            ps.append(step_toward(ps[-1], tgt) if eng.free else dict(tgt))
        c2rigs = [view_c2rig(p) for p in ps]
        frames, warps, zc_new, tm = eng.chunk(k, c2rigs, zc)
        if not eng.free and VIEW.cam()[1] != ver:      # view changed while generating: drop this chunk
            continue
        zc, cur = zc_new, ps[-1]
        ema = tm["total"] if ema is None else 0.7 * ema + 0.3 * tm["total"]
        STATS.update(status="running", fps=8 / ema, chunk_ms={n: round(t * 1000) for n, t in tm.items()},
                     cam={n: round(x, 1) for n, x in cur.items()})
        jpgs = []
        for j, (im, w) in enumerate(zip(frames, warps)):
            im = compose(im, w, c2rigs[j + 1], eng.meta["chase_f"], VIEW.car, VIEW.inset)
            cam = json.dumps({n: round(x, 1) for n, x in ps[j + 1].items()})
            jpgs.append((k + 1 + j, cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), cam))
        with OUT_COND:
            for fi, b, cam in jpgs:
                SEQ[0] += 1
                OUT.append((SEQ[0], fi, b, token, cam))
            while len(OUT) > 24:
                OUT.popleft()
            OUT_COND.notify_all()
        k += 8
        while len(OUT) > ahead_frames and (eng.free or VIEW.cam()[1] == ver) and not VIEW.pause:
            time.sleep(0.01)


GLOBAL = {}


def next_frame(after):
    """oldest frame with seq > after; playback pacing is done by the client."""
    with OUT_COND:
        end = time.time() + 5
        while True:
            for item in OUT:
                if item[0] > after:
                    # consume everything older than what is handed out, so the buffer drains as it plays
                    while OUT and OUT[0][0] < item[0]:
                        OUT.popleft()
                    return item
            if time.time() > end:
                return None
            OUT_COND.wait(0.5)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype, extra=None, code=200):
        self.send_response(code)
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
            elif u.path == "/stats":
                self.send(json.dumps(STATS).encode(), "application/json")
            elif u.path == "/set":
                VIEW.set(q)
                self.send(b"ok", "text/plain")
            elif u.path == "/next":
                item = next_frame(int(q.get("after", 0)))
                if item is None:
                    self.send(b"", "text/plain", code=204)
                else:
                    self.send(item[2], "image/jpeg", {"X-Seq": str(item[0]), "X-Frame": str(item[1]), "X-Buffered": str(len(OUT)), "X-View": item[3],
                                                      "X-Cam": item[4], "X-Stats": json.dumps(STATS)})
            elif u.path == "/preview":
                eng = GLOBAL.get("eng")
                p = clamp_view({n: float(q.get(n, VIEW.p[n])) for n in ("yaw", "pitch", "dist", "ahead")})
                j = int(q.get("frame", -1))
                c2rig = view_c2rig(p)
                img = eng.reproject(j, c2rig) if eng is not None else None
                if img is None:
                    self.send(b"", "text/plain", code=204); return
                img = compose(img, None, c2rig, eng.meta["chase_f"], VIEW.car, False)
                self.send(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), "image/jpeg",
                          {"X-Frame": str(j), "X-Cam": json.dumps(p)})
            elif u.path == "/surround":
                eng = GLOBAL.get("eng")
                if eng is None:
                    self.send(b"", "text/plain", code=204); return
                rgb = eng.rgb[int(q.get("i", 0)) % eng.N].cpu().numpy()
                order = [1, 0, 2, 3, 5, 4]   # FL F FR / BL B BR
                m = np.vstack([np.hstack([rgb[o] for o in order[:3]]), np.hstack([rgb[o] for o in order[3:]])])
                self.send(cv2.imencode(".jpg", cv2.resize(m[..., ::-1], (720, 264)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), "image/jpeg")
            else:
                self.send(b"not found", "text/plain", code=404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:      # noqa: BLE001
            print("ERR", repr(e), flush=True)
            self.send(repr(e).encode(), "text/plain", code=500)


def run_generator():
    try:
        generator()
    except Exception as e:          # noqa: BLE001
        import traceback
        traceback.print_exc()
        STATS["status"] = "error: " + repr(e)


if __name__ == "__main__":
    threading.Thread(target=run_generator, daemon=True).start()
    print(f"serving on http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
