"""Stylised low-poly ego car (Tesla-like fastback) drawn over generated frames, like the in-car Reality View.
Rig frame: x fwd, y left, z up, origin = rear axle on the ground. Camera OpenCV pinhole (c2rig 4x4).
Painter's algorithm with flat Lambert shading + a soft contact shadow; no scene depth needed (the ego is always in front of
whatever sits in its footprint)."""
import numpy as np
import cv2

L0, L1, HW = -1.02, 3.70, 0.94        # rear / front bumper x, half width (m)
WB, WR = 2.88, 0.34                   # wheelbase, wheel radius
BODY = [(L0, 0.30), (L0, 0.78), (L0 + 0.25, 0.98), (L0 + 0.75, 1.42), (1.85, 1.45), (2.55, 1.02), (L1 - 0.05, 0.80), (L1, 0.48), (L1 - 0.05, 0.30)]


def _mesh():
    V, Fc = [], []   # side profile extruded across the width, roof narrower (tumblehome)
    n = len(BODY)
    for side in (1, -1):
        for x, z in BODY:
            w = HW * (0.80 if z > 1.0 else 1.0)
            V.append((x, side * w, z))
    for i in range(n):                       # skin quads around the profile (segment 2 = rear glass, 4 = windscreen)
        j = (i + 1) % n
        Fc.append(([i, j, n + j, n + i], "glass" if i in (2, 4) else "body"))
    Fc.append((list(range(n)), "side")); Fc.append((list(range(2 * n - 1, n - 1, -1)), "side"))
    # windows on the sides: inset polygon on the greenhouse
    for side, off in ((1, 0), (-1, n)):
        win = [(L0 + 0.85, 1.36), (1.80, 1.39), (2.40, 1.06), (L0 + 0.45, 1.02)]
        base = len(V)
        for x, z in win: V.append((x, side * (HW * 0.80 + 0.012), z))
        Fc.append((list(range(base, base + 4)), "sidewin"))
    # wheels: 12-gon discs on both sides
    for cx in (0.0, WB):
        for side in (1, -1):
            base = len(V)
            for a in np.linspace(0, 2 * np.pi, 13)[:-1]:
                V.append((cx + WR * np.cos(a), side * (HW + 0.01), WR + WR * np.sin(a)))
            Fc.append((list(range(base, base + 12)), "tyre"))
    return np.array(V, np.float32), Fc


VERTS, FACES = _mesh()
COL = {"body": np.array([236, 236, 240.]), "side": np.array([226, 226, 232.]), "glass": np.array([40, 44, 52.]), "sidewin": np.array([46, 50, 58.]), "tyre": np.array([24, 24, 26.])}
LIGHT = np.array([0.4, 0.3, 0.86]); LIGHT /= np.linalg.norm(LIGHT)


def draw(img, c2rig, fx, fy):
    """img HxWx3 uint8 BGR (modified copy returned)."""
    H, W = img.shape[:2]; out = img.copy()
    w2c = np.linalg.inv(c2rig)
    # contact shadow: blurred dark ellipse on the ground under the car
    sh = np.zeros((H, W), np.float32); g = []
    for a in np.linspace(0, 2 * np.pi, 40):
        g.append(((L0 + L1) / 2 + (L1 - L0) / 2 * 1.12 * np.cos(a), HW * 1.35 * np.sin(a), 0.01, 1))
    q = (w2c @ np.array(g).T)[:3]
    if (q[2] > 0.1).all():
        pts = np.stack([W / 2 + fx * q[0] / q[2], H / 2 + fy * q[1] / q[2]], 1).astype(np.int32)
        cv2.fillPoly(sh, [pts], 0.55); sh = cv2.GaussianBlur(sh, (0, 0), max(2.0, W / 90))
        out = (out * (1 - sh[..., None])).astype(np.uint8)
    q = (w2c[:3, :3] @ VERTS.T).T + w2c[:3, 3]
    if (q[:, 2] < 0.05).any(): return out              # camera inside / behind the car: skip
    uv = np.stack([W / 2 + fx * q[:, 0] / q[:, 2], H / 2 + fy * q[:, 1] / q[:, 2]], 1)
    cam = c2rig[:3, 3]; polys = []
    for idx, kind in FACES:
        P = VERTS[idx]; ctr = P.mean(0)
        if kind in ("side", "sidewin", "tyre"):                 # lateral faces: outward = +-y
            nrm = np.array([0, np.sign(ctr[1]), 0.0])
        else:                                                   # skin quad: outward normal (profile winding gives inward)
            nrm = -np.cross(P[1] - P[0], P[2] - P[0])
            if np.linalg.norm(nrm) < 1e-9: continue
            nrm /= np.linalg.norm(nrm)
        if np.dot(nrm, cam - ctr) <= 0: continue                # back-face cull
        shade = 0.55 + 0.45 * abs(float(np.dot(nrm, LIGHT)))
        polys.append((q[idx, 2].mean() - (0.6 if kind in ("sidewin", "tyre") else 0), uv[idx], COL[kind] * shade))   # decals after their panel
    for _, p, c in sorted(polys, key=lambda t: -t[0]):
        cv2.fillPoly(out, [np.round(p * 4).astype(np.int32)], tuple(float(v) for v in c), lineType=cv2.LINE_AA, shift=2)
    return out
