"""Pixel-aligned geometric condition of the video student: the accumulated point cloud of the 6 rig cameras rendered
into a virtual camera (used offline by scripts/student/prepare_warp.py and live by viewer/live.py).

  1. dense metric depth of every rig image = DA3METRIC, scaled to the sparse lidar depth (median ratio) when available,
     with depth edges removed (they produce flying pixels)
  2. points of the current frame are splatted first (z-buffer); points of past frames only fill pixels the current
     frame leaves empty, so moving objects do not leave ghost trails
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

CAMS = ["F", "FL", "FR", "BL", "BR", "B"]
PAST = (2, 4, 6, 8, 12, 16, 20)     # up to 2 s back: the road under / beside the car was seen by the front cameras earlier
MAX_DEPTH = 80.0


def read_inputs(d, k):
    """6 rig images of frame k, RGB uint8 [6, H, W, 3]."""
    return np.stack([cv2.imread(f"{d}/inputs/{k:04d}_{c}.jpg")[..., ::-1].copy() for c in CAMS])


def frame_depth(da3, rgbs, d, k, meta, use_lidar=True):
    """[6, H, W] float32 metric depth (0 = invalid) of one frame."""
    D, _ = da3(list(rgbs), [c["f"] for c in meta["rig"]])
    out = np.zeros((6, meta["in_h"], meta["in_w"]), np.float32)
    for c in range(6):
        s = 1.0
        if use_lidar:
            lid = cv2.imread(f"{d}/inputs/{k:04d}_{CAMS[c]}_d.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / meta["depth_scale"]
            m = (lid > 1) & (lid < 60) & (D[c] > 0)
            s = np.median(lid[m] / D[c][m]) if m.sum() > 50 else 1.0
        z = D[c] * s
        mx = cv2.dilate(z, np.ones((3, 3))); mn = cv2.erode(z, np.ones((3, 3)))
        z[(mx - mn) > 0.06 * z] = 0
        z[z > MAX_DEPTH] = 0
        out[c] = z
    return out


def dense_depth(da3, d, meta):
    """[N, 6, H, W] float32 metric depth of every frame of a teacher scene."""
    return np.stack([frame_depth(da3, read_inputs(d, fr["k"]), d, fr["k"], meta) for fr in meta["frames"]])


class Splatter:
    def __init__(self, meta, dev="cuda"):
        self.dev = dev
        H, W = meta["in_h"], meta["in_w"]
        v, u = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        self.rays = []
        for c in meta["rig"]:
            r = torch.stack([(u - c["cx"]) / c["f"], (v - c["cy"]) / c["f"], torch.ones_like(u, dtype=torch.float32)], -1).float()
            self.rays.append(r.view(-1, 3))
        self.c2rig = [torch.tensor(c["c2rig"], device=dev, dtype=torch.float32) for c in meta["rig"]]
        self.tw, self.th, self.tf = meta["chase_w"], meta["chase_h"], meta["chase_f"]

    def points(self, depth_k, rgb_k, e2w):
        """world points + colors of the 6 cameras of one frame (torch, on device)."""
        P, C = [], []
        for c in range(6):
            z = depth_k[c].reshape(-1)
            m = z > 0
            p = self.rays[c][m] * z[m, None]
            T = e2w @ self.c2rig[c]
            P.append(p @ T[:3, :3].T + T[:3, 3]); C.append(rgb_k[c].reshape(-1, 3)[m])
        return torch.cat(P), torch.cat(C)

    def render_t(self, layers, c2w):
        """layers: [(P, C), ...] in priority order -> HxWx3 uint8 RGB tensor, HxW bool tensor (on device)."""
        W, H = self.tw, self.th
        w2c = torch.linalg.inv(c2w)
        img = torch.zeros(H * W, 3, device=self.dev, dtype=torch.uint8)
        filled = torch.zeros(H * W, device=self.dev, dtype=torch.bool)
        for P, C in layers:
            p = P @ w2c[:3, :3].T + w2c[:3, 3]
            ok = p[:, 2] > 0.3
            p, col = p[ok], C[ok]
            u = p[:, 0] / p[:, 2] * self.tf + W / 2; v = p[:, 1] / p[:, 2] * self.tf + H / 2
            zb = torch.full((H * W,), float("inf"), device=self.dev)
            idxs, zs, cs = [], [], []
            for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):          # 2x2 splat closes most sampling holes
                ui = (u + du - 0.5).floor().long(); vi = (v + dv - 0.5).floor().long()
                m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                idx = vi[m] * W + ui[m]
                m2 = ~filled[idx]
                idxs.append(idx[m2]); zs.append(p[m, 2][m2]); cs.append(col[m][m2])
            idx, z, c = torch.cat(idxs), torch.cat(zs), torch.cat(cs)
            zb.scatter_reduce_(0, idx, z, "amin")
            win = z <= zb[idx]
            img[idx[win]] = c[win]
            filled |= torch.isfinite(zb)
        return img.view(H, W, 3), filled.view(H, W)

    def render_fast(self, layers, c2w, with_depth=False):
        """Same result as render_t in one pass over all layers: the z-buffer key is layer * 1000 + depth, so a pixel
        hit by an earlier (higher priority) layer is never taken by a later one, and depth decides inside a layer.
        with_depth: also return the HxW camera depth of the winning points (0 where empty)."""
        W, H = self.tw, self.th
        w2c = torch.linalg.inv(c2w)
        P = torch.cat([p for p, _ in layers]); C = torch.cat([c for _, c in layers])
        lay = torch.cat([torch.full((len(p),), float(i), device=self.dev) for i, (p, _) in enumerate(layers)])
        p = P @ w2c[:3, :3].T + w2c[:3, 3]
        ok = p[:, 2] > 0.3
        p, C, lay = p[ok], C[ok], lay[ok]
        u = p[:, 0] / p[:, 2] * self.tf + W / 2 - 0.5; v = p[:, 1] / p[:, 2] * self.tf + H / 2 - 0.5
        key = lay * 1000.0 + p[:, 2]                    # depth < MAX_DEPTH (80) << 1000
        u0, v0 = u.floor().long(), v.floor().long()
        idxs, keys, src = [], [], []
        ar = torch.arange(len(p), device=self.dev)
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ui, vi = u0 + du, v0 + dv
            m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            idxs.append(vi[m] * W + ui[m]); keys.append(key[m]); src.append(ar[m])
        idx, key, src = torch.cat(idxs), torch.cat(keys), torch.cat(src)
        zb = torch.full((H * W,), float("inf"), device=self.dev)
        zb.scatter_reduce_(0, idx, key, "amin")
        win = key <= zb[idx]
        img = torch.zeros(H * W, 3, device=self.dev, dtype=torch.uint8)
        img[idx[win]] = C[src[win]]
        filled = torch.isfinite(zb)
        if with_depth:
            z = torch.where(filled, torch.remainder(zb, 1000.0), torch.zeros_like(zb))
            return img.view(H, W, 3), filled.view(H, W), z.view(H, W)
        return img.view(H, W, 3), filled.view(H, W)

    def unproject_view(self, img, depth, c2w, far=60.0):
        """A rendered view (HxWx3 uint8, HxW depth, 0 = unknown) back to world points + colors; unknown pixels (sky,
        regions no rig camera saw) are put at `far` so they move like distant background."""
        H, W = depth.shape
        v, u = torch.meshgrid(torch.arange(H, device=self.dev), torch.arange(W, device=self.dev), indexing="ij")
        z = torch.where(depth > 0, depth, torch.full_like(depth, far)).reshape(-1)
        p = torch.stack([(u.reshape(-1) + 0.5 - W / 2) / self.tf * z, (v.reshape(-1) + 0.5 - H / 2) / self.tf * z, z], -1)
        return p @ c2w[:3, :3].T + c2w[:3, 3], img.reshape(-1, 3)

    def render(self, layers, c2w):
        """numpy version of render_fast (render_t is the reference, layer-by-layer implementation)."""
        img, m = self.render_fast(layers, c2w)
        return img.cpu().numpy(), m.cpu().numpy()


def fill_holes(img, mask, levels=7):
    """Push-pull hole filling on the GPU: HxWx3 uint8 + HxW bool (valid) -> HxWx3 uint8 where every empty pixel takes
    the average color of the nearest valid ones (normalized convolution over an image pyramid)."""
    x = img.permute(2, 0, 1)[None].float(); m = mask[None, None].float()
    pyr = [(x * m, m)]
    for _ in range(levels):                                     # push: average valid pixels into coarser levels
        c, w = pyr[-1]
        pyr.append((F.avg_pool2d(c, 2, ceil_mode=True), F.avg_pool2d(w, 2, ceil_mode=True)))
    c, w = pyr[-1]
    fill = c / w.clamp(min=1e-6)
    for c, w in reversed(pyr[:-1]):                              # pull: fill each level's holes from the coarser one
        up = F.interpolate(fill, size=c.shape[-2:], mode="bilinear", align_corners=False)
        fill = torch.where(w > 1e-6, c / w.clamp(min=1e-6), up)
    out = torch.where(m > 0, x, fill)
    return out[0].permute(1, 2, 0).round().clamp(0, 255).byte()


def pack_mask(masks):
    """[9, H, W] bool (numpy or tensor) -> [4, 3, H/8, W/8] as in Wan I2V (first frame repeated 4x, then groups of 4)."""
    m = (torch.as_tensor(np.stack(masks)) if not torch.is_tensor(masks) else masks).float()[None]   # 1, 9, H, W
    m = F.avg_pool2d(m, 8)[0]                                                   # 9, h, w
    m = torch.cat([m[:1].repeat(4, 1, 1), m[1:]])                               # 12, h, w
    return m.view(3, 4, *m.shape[1:]).transpose(0, 1).contiguous()             # 4, 3, h, w
