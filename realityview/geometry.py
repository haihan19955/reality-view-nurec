"""Camera geometry: lifting depth to points, z-buffer splatting, Plücker rays, the ego zone and orbit cameras.

Rig / ego frame: x forward, y left, z up, origin on the ground at the rear axle.
Cameras: OpenCV (x right, y down, z forward), pinhole, principal point at the image centre.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

IN_W, IN_H = 640, 352                 # rendered surround inputs / targets
TW, TH = 384, 208                     # model target resolution (latent 48x26 -> 24x13 patch tokens)
TF_SRC = 320.0                        # target focal at 640x352 (90 deg horizontal FoV)
TFX, TFY = TF_SRC * TW / IN_W, TF_SRC * TH / IN_H
LW, LH = TW // 8, TH // 8             # latent size
EGO_ZONE = (-2.5, 6.4, -3.5, 3.5)     # x0, x1, y0, y1 on the ground: ego footprint + 1.5-2.5 m margin
LOOK_AT = (1.4, 0.0, 0.6)             # orbit cameras look at the middle of the car


def lift(depths, colors, c2rig, fs):
    """depths [V,H,W] metres (0 = invalid/sky), colors [V,H,W,3] uint8, c2rig [V,4,4], fs [V] -> points [N,3] (rig), colors [N,3]."""
    dev = depths.device
    _, H, W = depths.shape
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    P, C = [], []
    for k in range(len(depths)):
        d = depths[k]
        m = (d > 0.3) & (d < 150)
        f = fs[k]
        p = torch.stack([(u - W / 2 + 0.5) / f * d, (v - H / 2 + 0.5) / f * d, d, torch.ones_like(d)], -1)[m]
        P.append((c2rig[k] @ p.T).T[:, :3])
        C.append(colors[k][m])
    return torch.cat(P), torch.cat(C)


def splat(pts, col, c2rig, W=TW, H=TH, fx=TFX, fy=TFY):
    """Z-buffered point splat into a pinhole camera -> image [H,W,3] uint8, coverage mask [H,W] bool."""
    dev = pts.device
    w2c = torch.linalg.inv(c2rig)
    q = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    ok = q[:, 2] > 0.2
    q, c = q[ok], col[ok]
    u = (W / 2 + fx * q[:, 0] / q[:, 2]).floor().long()
    v = (H / 2 + fy * q[:, 1] / q[:, 2]).floor().long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, c = u[ok], v[ok], q[ok, 2], c[ok]
    idx = v * W + u
    zb = torch.full((W * H,), float("inf"), device=dev).scatter_reduce(0, idx, z, "amin")
    win = z <= zb[idx] + 1e-3
    img = torch.zeros(W * H, 3, dtype=torch.uint8, device=dev)
    img[idx[win]] = c[win]
    return img.view(H, W, 3), torch.isfinite(zb).view(H, W)


def fill_small_holes(img, msk, levels=3):
    """Pull-push fill of holes up to ~2**levels px; large holes (e.g. under the ego car) stay empty.
    img [H,W,3] uint8, msk [H,W] bool -> filled image, filled mask."""
    x = img.permute(2, 0, 1)[None].float() * msk
    w = msk[None, None].float()
    pyr = [(x, w)]
    for _ in range(levels):
        x = F.avg_pool2d(x, 2) * 4
        w = F.avg_pool2d(w, 2) * 4
        pyr.append((x, w))
    col = pyr[-1][0] / pyr[-1][1].clamp(min=1e-6)
    val = pyr[-1][1] > 0
    for x, w in reversed(pyr[:-1]):
        col = F.interpolate(col, scale_factor=2, mode="bilinear", align_corners=False)
        val = F.interpolate(val.float(), scale_factor=2, mode="nearest") > 0
        own = w > 0
        col = torch.where(own, x / w.clamp(min=1e-6), col)
        val = own | val
    # keep fills only near a real point, so big holes do not bleed
    near = F.max_pool2d(msk[None, None].float(), 2 ** levels + 1, 1, 2 ** (levels - 1)) > 0
    val = val & near
    out = torch.where(val, col, torch.zeros_like(col))
    return out[0].permute(1, 2, 0).round().clamp(0, 255).byte(), val[0, 0]


def geo_render(pts, col, c2rig):
    """Splat + small-hole fill at the model resolution."""
    img, msk = splat(pts, col, c2rig)
    return fill_small_holes(img, msk)


def composite_holes(real, gen, mask, grow=7, feather=21):
    """Keep real pixels where the cameras saw, use the generated image only in the holes, with a feathered seam.
    real, gen [3,H,W] float 0..255; mask [H,W] bool (covered) -> [3,H,W]."""
    hole = 1 - mask.float()[None, None]
    hole = F.max_pool2d(hole, grow, 1, grow // 2)
    hole = F.avg_pool2d(hole, feather, 1, feather // 2, count_include_pad=False)[0]
    return real * (1 - hole) + gen * hole


def rays_cam(H, W, fx, fy, dev):
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    return torch.stack([(u + 0.5 - W / 2) / fx, (v + 0.5 - H / 2) / fy, torch.ones_like(u)], -1)


def plucker(rel, ph=TH // 16, pw=TW // 16):
    """rel [B,4,4] camera->rig -> [B, ph*pw, 6] Plücker rays at patch centres (patch = 2x2 latent = 16x16 px)."""
    d = rays_cam(ph, pw, TFX / 16, TFY / 16, rel.device).view(-1, 3)
    d = d / d.norm(dim=-1, keepdim=True)
    dw = torch.einsum("bij,nj->bni", rel[:, :3, :3], d)
    o = rel[:, :3, 3]
    mom = torch.cross(o[:, None].expand_as(dw), dw, dim=-1)
    return torch.cat([dw, mom / 10.0], -1)


def ego_zone_mask(rel, H=LH, W=LW):
    """[B,H,W] True where the pixel's ray hits the ground inside EGO_ZONE
    (NuRec is unreliable right next to the ego; the CG car covers it anyway)."""
    d = rays_cam(H, W, TFX / 8, TFY / 8, rel.device).view(-1, 3)
    dw = torch.einsum("bij,nj->bni", rel[:, :3, :3], d)
    o = rel[:, :3, 3][:, None]
    s = -o[..., 2] / dw[..., 2]
    hit = (dw[..., 2] < -1e-3) & (s > 0)
    p = o + s[..., None] * dw
    x0, x1, y0, y1 = EGO_ZONE
    inside = hit & (p[..., 0] > x0) & (p[..., 0] < x1) & (p[..., 1] > y0) & (p[..., 1] < y1)
    return inside.view(-1, H, W)


def look_at(pos, target):
    """camera->rig 4x4 (OpenCV camera) at pos looking at target, rig z up."""
    fwd = np.asarray(target, np.float64) - np.asarray(pos, np.float64)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = np.stack([right, down, fwd], 1)
    M[:3, 3] = pos
    return M


def chase_camera(back, height, yaw=0.0, look_ahead=8.0, center_x=1.4):
    """Rig-fixed version of the teacher's chase camera (scripts/nurec/render_teacher.py): `back` m behind the rear axle,
    `height` m up, looking at a point `look_ahead` m ahead at 0.6 m. yaw (deg, left +) swings the whole rig around the
    car's vertical axis: 0 = behind, 90 = from the right side looking left, 180 = in front looking back."""
    pos, target = np.array([-back, 0.0, height]), np.array([look_ahead, 0.0, 0.6])
    a = math.radians(yaw)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    c = np.array([center_x, 0.0, 0.0])
    return look_at(R @ (pos - c) + c, R @ (target - c) + c)


def orbit_camera(yaw, pitch, dist, target=LOOK_AT):
    """Orbit camera around the ego. yaw 180 = behind the car, pitch = elevation (degrees), dist in metres."""
    y, p = math.radians(yaw), math.radians(pitch)
    tx, ty, tz = target
    pos = np.array([tx + dist * math.cos(p) * math.cos(y), ty + dist * math.cos(p) * math.sin(y), tz + dist * math.sin(p)])
    return look_at(pos, target)
