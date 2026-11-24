"""DrivingForward (SF mode, nuScenes weights) as a library on NuRec surround frames: 6 views -> per-pixel 3D Gaussians -> render any camera.
Bypasses the nuScenes dataloader / losses (no LPIPS / torchvision downloads). Rig/ego frame: x fwd, y left, z up (same as nuScenes ego)."""
import os, sys, types, yaml, numpy as np, torch, torch.nn.functional as F
from realityview import paths
DF = paths.model("DrivingForward"); sys.path.insert(0, DF)

# pytorch3d is only used for axis_angle_to_matrix (pose net); provide a tiny shim instead of the 2020-era package
if "pytorch3d" not in sys.modules:
    p3 = types.ModuleType("pytorch3d"); tr = types.ModuleType("pytorch3d.transforms")
    def axis_angle_to_matrix(aa):
        th = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8); k = aa / th; K = torch.zeros(*aa.shape[:-1], 3, 3, device=aa.device, dtype=aa.dtype)
        K[..., 0, 1], K[..., 0, 2], K[..., 1, 0], K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 2], k[..., 1], k[..., 2], -k[..., 0], -k[..., 1], k[..., 0]
        s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
        return torch.eye(3, device=aa.device, dtype=aa.dtype) + s * K + (1 - c) * K @ K
    tr.axis_angle_to_matrix = axis_angle_to_matrix; p3.transforms = tr
    sys.modules["pytorch3d"] = p3; sys.modules["pytorch3d.transforms"] = tr

def get_relcam(cameras):   # utils.misc.get_relcam, inlined (DF's utils package pulls tensorboardX)
    L = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    R = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]}
    idx = [L.index(c) for c in cameras]; return {i: [c for c in R[i] if c in idx] for i in idx}
# import models.gaussian without running models/__init__ (which pulls the nuScenes dataloader + LPIPS losses)
_m = types.ModuleType("models"); _m.__path__ = [f"{DF}/models"]; sys.modules.setdefault("models", _m)
from network.depth_network import DepthNetwork
from models.gaussian import GaussianNetwork, depth2pc, focal2fov, getProjectionMatrix, rotate_sh
from models.gaussian.gaussian_renderer import render
from einops import rearrange


class DFRunner:
    def __init__(self, weights=f"{DF}/weights_SF", H=352, W=640):
        cfg = yaml.safe_load(open(f"{DF}/configs/nuscenes/main.yaml"))
        cfg["data"]["num_cams"] = 6; cfg["data"]["rel_cam_list"] = get_relcam(cfg["data"]["cameras"])
        cfg["model"].update(mode="eval", novel_view_mode="SF", weights_init=False)
        cfg["training"]["batch_size"] = 1
        self.cfg, self.H, self.W = cfg, H, W
        self.min_d, self.max_d, self.fscale = cfg["training"]["min_depth"], cfg["training"]["max_depth"], cfg["training"]["focal_length_scale"]
        self.depth_net = DepthNetwork(cfg).cuda().eval(); self.gs_net = GaussianNetwork(rgb_dim=3, depth_dim=1).cuda().eval()
        for net, n in [(self.depth_net, "depth_net"), (self.gs_net, "gs_net")]:
            sd = torch.load(f"{weights}/{n}.pth", map_location="cuda"); md = net.state_dict()
            miss = [k for k in md if k not in sd]; md.update({k: v for k, v in sd.items() if k in md}); net.load_state_dict(md)
            assert not miss, f"{n} missing {len(miss)} keys e.g. {miss[:3]}"

    @torch.no_grad()
    def reconstruct(self, imgs_bgr, fs, c2rig):
        """imgs_bgr: 6 HxWx3 uint8 (F,FL,FR,BL,BR,B); fs: focal px; c2rig: 6 4x4 camera->ego. Returns gaussians + per-view depth."""
        H, W = self.H, self.W
        col = torch.stack([torch.from_numpy(i[..., ::-1].copy()).permute(2, 0, 1).float() / 255 for i in imgs_bgr])[None].cuda()   # 1,6,3,H,W
        ext = torch.from_numpy(np.stack(c2rig)).float()[None].cuda(); ext_inv = torch.inverse(ext)
        inputs = {("color", 0, 0): col, ("color_aug", 0, 0): col, "extrinsics": ext, "extrinsics_inv": ext_inv,
                  "mask": torch.ones(1, 6, 1, H, W, device="cuda")}
        for s in range(4):
            K = torch.zeros(1, 6, 4, 4, device="cuda")
            for c, f in enumerate(fs):
                K[0, c, 0, 0] = K[0, c, 1, 1] = f / 2 ** s; K[0, c, 0, 2] = W / 2 / 2 ** s; K[0, c, 1, 2] = H / 2 / 2 ** s; K[0, c, 2, 2] = K[0, c, 3, 3] = 1
            inputs[("K", s)] = K; inputs[("inv_K", s)] = torch.inverse(K)
        out = self.depth_net(inputs)
        G = dict(xyz=[], rot=[], scale=[], op=[], sh=[]); depths = []
        for c in range(6):
            o = out[("cam", c)]; Kc = inputs[("K", 0)][:, c]
            disp = F.interpolate(o[("disp", 0)], [H, W], mode="bilinear", align_corners=False)
            depth = 1 / (1 / self.max_d + (1 / self.min_d - 1 / self.max_d) * disp) * Kc[:, 0:1, 0:1].unsqueeze(2) / self.fscale
            depths.append(depth[0, 0].cpu().numpy())
            xyz = depth2pc(depth, ext_inv[:, c], Kc)[0]
            rot, scale, op, sh = self.gs_net(col[:, c], depth, o[("img_feat", 0, 0)])
            # e3nn>=0.5 builds wigner_D generators on the CPU -> rotate there (TODO: cache D per camera for speed)
            sh = rotate_sh(sh.cpu(), rearrange(ext[:, c, :3, :3].cpu(), "k i j -> k () () () i j")[..., None, :, :]).cuda()
            G["xyz"].append(xyz); G["rot"].append(rot[0].permute(1, 2, 0).reshape(-1, 4)); G["scale"].append(scale[0].permute(1, 2, 0).reshape(-1, 3))
            G["op"].append(op[0].permute(1, 2, 0).reshape(-1, 1)); G["sh"].append(rearrange(sh[0], "p srf r xyz d_sh -> (p srf r) d_sh xyz").contiguous())
        return {k: torch.cat(v) for k, v in G.items()}, depths

    @torch.no_grad()
    def render(self, G, c2rig, f, H=None, W=None, bg=(0, 0, 0)):
        H, W = H or self.H, W or self.W
        K = torch.tensor([[f, 0, W / 2, 0], [0, f, H / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cuda")
        e2c = torch.inverse(torch.from_numpy(np.asarray(c2rig)).float().cuda())
        proj = getProjectionMatrix(znear=0.01, zfar=self.max_d, K=K, h=H, w=W).transpose(0, 1).cuda()
        wv = e2c.transpose(0, 1); full = (wv[None].bmm(proj[None]))[0]
        img = render(novel_FovX=focal2fov(K[0, 0], W), novel_FovY=focal2fov(K[1, 1], H), novel_height=H, novel_width=W,
                     novel_world_view_transform=wv, novel_full_proj_transform=full, novel_camera_center=wv.inverse()[3, :3],
                     pts_xyz=G["xyz"], pts_rgb=None, rotations=G["rot"], scales=G["scale"], opacity=G["op"], shs=G["sh"], bg_color=list(bg))
        return img.clamp(0, 1)

    def render_bgr(self, G, c2rig, f):
        """uint8 BGR image + coverage mask (pixels that do not depend on the background colour)."""
        a = self.render(G, c2rig, f, bg=(0, 0, 0)); b = self.render(G, c2rig, f, bg=(1, 1, 1))
        cov = ((b - a).abs().max(0)[0] < 0.1).cpu().numpy()
        img = (a.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
        return img, cov


if __name__ == "__main__":   # smoke test on one NuRec frame: re-render the inputs + 2 targets
    import json, glob, cv2, time
    d = sorted(glob.glob(os.path.join(paths.NUREC_DATA, "*", "meta.json")))[0]; meta = json.load(open(d)); d = os.path.dirname(d)
    fr = meta["frames"][40]; ins = [v for v in fr["views"] if v["kind"] == "in"]; tg = [v for v in fr["views"] if v["kind"] != "in"]
    r = DFRunner()
    imgs = [cv2.imread(f"{d}/{v['file']}.jpg") for v in ins]
    for _ in range(2):
        torch.cuda.synchronize(); t0 = time.time()
        G, dep = r.reconstruct(imgs, [v["f"] for v in ins], [np.array(v["c2rig"]) for v in ins]); torch.cuda.synchronize(); t1 = time.time()
        out, cov = r.render_bgr(G, np.array(tg[0]["c2rig"]), tg[0]["f"]); torch.cuda.synchronize(); t2 = time.time()
    print(f"gaussians {len(G['xyz'])}  reconstruct {1000 * (t1 - t0):.0f} ms  render(x2 bg) {1000 * (t2 - t1):.0f} ms")
    for v, D in zip(ins, dep):
        L = cv2.imread(f"{d}/{v['file']}_d.png", -1).astype(np.float32) / 256; m = (L > 1) & (L < 30)
        print(v["name"], "median DF/lidar (1-30 m)", round(float(np.median(D[m] / L[m])), 3))
    rows = []
    for v in [ins[0], ins[3]] + tg[:3]:
        o, c = r.render_bgr(G, np.array(v["c2rig"]), v["f"]); gt = cv2.imread(f"{d}/{v['file']}.jpg")
        rows.append(np.hstack([gt, o]))
    os.makedirs(paths.OUTPUTS, exist_ok=True)
    cv2.imwrite(os.path.join(paths.OUTPUTS, "drivingforward_smoke.jpg"), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
