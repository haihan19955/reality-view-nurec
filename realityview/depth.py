"""Metric depth from Depth Anything 3 (DA3METRIC-LARGE)."""
import cv2
import numpy as np


class MetricDepth:
    """metric depth (m) = focal of the processed image (px) * network output / 300.  Also returns a sky mask."""

    def __init__(self, path, device="cuda", process_res=504):
        from depth_anything_3.api import DepthAnything3
        self.net = DepthAnything3.from_pretrained(path).to(device).eval()
        self.process_res = process_res

    def __call__(self, rgbs, fs, zero_sky=True):
        """rgbs: list of HxWx3 uint8 RGB, fs: focal (px) per image -> (depths [HxW float32], sky masks [HxW bool])."""
        H, W = rgbs[0].shape[:2]
        p = self.net.inference(list(rgbs), process_res=self.process_res)
        w = p.depth.shape[2]
        depths, skies = [], []
        for i, f in enumerate(fs):
            D = cv2.resize(p.depth[i] * (f * w / W) / 300.0, (W, H))
            sky = cv2.resize(p.sky[i].astype(np.float32), (W, H)) > 0.5
            if zero_sky:
                D[sky] = 0      # sky is background, not geometry
            depths.append(D)
            skies.append(sky)
        return depths, skies
