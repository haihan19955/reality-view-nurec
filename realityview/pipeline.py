"""End-to-end Reality View renderer: 6 surround images -> any orbit view.

    rv = RealityView(checkpoint)
    state = rv.prepare(rgbs, focals, c2rig, tokens)     # once per frame: DA3 depth -> point cloud, camera context
    bgr = rv.render(state, orbit_camera(180, 22, 7))    # per view: splat -> MeanFlow 1-step -> composite -> ego car
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import ego_car, paths
from .depth import MetricDepth
from .geometry import LH, LW, TFX, TFY, composite_holes, geo_render, lift, plucker
from .latents import decode, encode, load_vae
from .model import load_checkpoint, sample


class RealityView:
    def __init__(self, checkpoint, device="cuda", depth_model=None):
        self.device = device
        self.model, ck = load_checkpoint(checkpoint, device)
        self.step, self.val_scenes = ck["step"], ck.get("val_sc", [])
        self.vae = load_vae(device)
        self.depth = MetricDepth(depth_model or paths.model("DA3METRIC-LARGE"), device)

    @torch.no_grad()
    def prepare(self, rgbs, fs, c2rig, tokens):
        """rgbs: 6 HxWx3 uint8 RGB (F, FL, FR, BL, BR, B); fs: focals; c2rig: 6 4x4; tokens: [6,112,256] PCA'd DINOv2 tokens."""
        dev = self.device
        depths, _ = self.depth(rgbs, fs)
        pts, col = lift(torch.tensor(np.stack(depths), device=dev), torch.tensor(np.stack(rgbs), device=dev),
                        torch.tensor(np.stack(c2rig), dtype=torch.float32, device=dev), fs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = self.model.context(torch.as_tensor(np.asarray(tokens, np.float32), device=dev)[None])
        return {"pts": pts, "col": col, "ctx": ctx}

    @torch.no_grad()
    def render(self, state, c2rig, geometry=True, composite=True, car=True, steps=1, seed=0, geometry_only=False):
        """c2rig: 4x4 OpenCV camera -> rig. Returns a 384x208 BGR uint8 image."""
        dev = self.device
        rel = torch.as_tensor(c2rig, dtype=torch.float32, device=dev)
        img, msk = geo_render(state["pts"], state["col"], rel)
        if geometry_only:
            out = img.cpu().numpy()[..., ::-1]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if geometry:
                    g = encode(self.vae, img[None])
                    m = (F.avg_pool2d(msk.float()[None, None], 8) > 0.5).float()
                else:
                    g = torch.zeros(1, 4, LH, LW, device=dev)
                    m = torch.zeros(1, 1, LH, LW, device=dev)
                noise = torch.randn(1, 4, LH, LW, device=dev, generator=torch.Generator(device=dev).manual_seed(int(seed)))
                z = sample(self.model, state["ctx"], plucker(rel[None]), g.float(), m, steps=steps, noise=noise)
            x = decode(self.vae, z.float())[0]
            if geometry and composite:       # real pixels where the cameras saw, generated only in the holes
                x = composite_holes(img.permute(2, 0, 1).float(), x, msk)
            out = x.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()[..., ::-1]
        out = np.ascontiguousarray(out)
        if car:
            out = ego_car.draw(out, np.asarray(c2rig), TFX, TFY)
        return out
