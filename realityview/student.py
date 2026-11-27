"""Video student: Wan2.1-1.3B DiT + LoRA, conditioned on surround-camera features, the previous frame and ego motion.

With meanflow=True the net is distilled (scripts/student/distill.py) to predict the average velocity over an interval,
so a chunk takes 1-2 steps instead of ~30.

One call generates a chunk of 3 Wan latent frames = 9 video frames at 832x480:
    latent 0      = the previous output frame (context, lightly noised, its own timestep)
    latents 1, 2  = the 8 new frames (denoised)
Per-token timesteps (the Wan 2.2 ti2v path in diffusers) let the context frame carry a different noise level
than the frames being generated, which is the "previous frame + noise" recipe.

With geo=True the main condition is pixel-aligned: the accumulated point cloud of the 6 rig cameras rendered into the
chase camera (scripts/student/prepare_warp.py), VAE-encoded and concatenated to the noisy latent together with its
observed mask, in the Wan I2V channel layout. The model then only has to fill holes, clean and keep time consistency.

The text cross-attention of Wan is reused for conditioning: its context tokens are
    DINOv2-L tokens of the 6 rig cameras at a few timestamps of the chunk (+ camera / time embeddings)
    ego-motion tokens (rig pose of each new frame relative to the context frame, chase-camera config)
"""
import torch
import torch.nn as nn

LAT_T, LAT_H, LAT_W = 3, 60, 104
PATCH_TOKENS = (LAT_H // 2) * (LAT_W // 2)      # Wan patch size (1, 2, 2)
FEAT_TIMES = (4, 8)                             # frame offsets in the chunk whose camera features are used
N_EGO_TOK = 4
EGO_DIM = 8 * 4 + 2                             # (dx, dy, dz, dyaw) for 8 new frames + (back, height)
GEO_CH = 4 + 16                                 # observed mask (Wan I2V packing) + VAE latent of the warped observation
LORA_TARGETS =["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]


class Student(nn.Module):
    def __init__(self, wan_path, rank=64, feat_dim=1024, n_cams=6, geo=False, geo_init=0.0, meanflow=False, ego_dim=EGO_DIM):
        super().__init__()
        self.geo, self.meanflow, self._h = geo, meanflow, None
        from diffusers import WanTransformer3DModel
        from peft import LoraConfig
        self.dit = WanTransformer3DModel.from_pretrained(wan_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.dit.requires_grad_(False)
        self.dit.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=LORA_TARGETS))
        self.dit.condition_embedder.text_embedder.requires_grad_(True)
        ctx = self.dit.config.text_dim
        self.feat = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, ctx))
        self.cam_emb = nn.Parameter(torch.zeros(n_cams, ctx))
        self.time_emb = nn.Parameter(torch.zeros(len(FEAT_TIMES), ctx))
        self.ego = nn.Sequential(nn.Linear(ego_dim, 1024), nn.SiLU(), nn.Linear(1024, N_EGO_TOK * ctx))
        nn.init.normal_(self.cam_emb, std=0.02)
        nn.init.normal_(self.time_emb, std=0.02)
        if geo:
            # Wan I2V layout: [noisy latent 16 | observed mask 4 | warped-observation latent 16]; new inputs start at zero
            old = self.dit.patch_embedding
            new = nn.Conv3d(16 + GEO_CH, old.out_channels, old.kernel_size, old.stride)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :16] = old.weight.float(); new.bias.copy_(old.bias.float())
                # the warp latent lives in the same VAE space as the noisy latent: start by reading it the same way
                # (zero init left it nearly unused after 3k steps: |W_warp| 1.2 vs |W_x| 10.4)
                new.weight[:, 20:] = old.weight.float() * geo_init
            self.dit.patch_embedding = new
            self.dit.register_to_config(in_channels=16 + GEO_CH)
        if meanflow:
            # MeanFlow: the net predicts the average velocity over [r, t]. The interval h = t - r gets its own timestep
            # embedding, added to Wan's time embedding; zero-init output so h is ignored at the start (u = v at h = 0).
            import copy
            ce = self.dit.condition_embedder
            self.h_emb = copy.deepcopy(ce.time_embedder).requires_grad_(True)
            nn.init.zeros_(self.h_emb.linear_2.weight); nn.init.zeros_(self.h_emb.linear_2.bias)

            def add_h(mod, inp, out):
                if self._h is None:
                    return out
                hp = ce.timesteps_proj(self._h.flatten()).view(*inp[0].shape)
                return out + self.h_emb(hp.float()).to(out.dtype)
            ce.time_embedder.register_forward_hook(add_h)
        for p in self.trainable():
            p.data = p.data.float()

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names}

    def context(self, feats, ego):
        """feats [B, T, 6, 112, 1024], ego [B, EGO_DIM] -> [B, T*6*112 + N_EGO_TOK, text_dim]"""
        B, T, C, N, _ = feats.shape
        f = self.feat(feats.float()) + self.cam_emb[None, None, :, None] + self.time_emb[None, :, None, None]
        e = self.ego(ego.float()).view(B, N_EGO_TOK, -1)
        return torch.cat([f.reshape(B, T * C * N, -1), e], 1)

    def forward(self, x, t_frames, feats, ego, geo=None, h_frames=None):
        """x [B,16,3,60,104] noisy latents, t_frames [B,3] in [0,1] per latent frame,
        geo [B,20,3,60,104] = observed mask (4) + warped observation latent (16) -> velocity (eps - x0).
        meanflow: h_frames [B,3] = t - r per latent frame -> average velocity over [r, t] (h = 0: instantaneous)."""
        ctx = self.context(feats, ego)
        ts = (t_frames * 1000).repeat_interleave(PATCH_TOKENS, dim=1)
        if self.meanflow:
            h = torch.zeros_like(t_frames) if h_frames is None else h_frames
            self._h = (h * 1000).repeat_interleave(PATCH_TOKENS, dim=1)
        if self.geo:
            x = torch.cat([x, geo.to(x)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.dit(hidden_states=x.bfloat16(), timestep=ts, encoder_hidden_states=ctx, return_dict=False)[0].float()


CAM_FRAMES = (0, 4, 8)                          # chunk frames whose camera pose (c2rig) is given (free-view models)
EGO_DIM_CAM = 8 * 4 + 6 * len(CAM_FRAMES)       # (dx, dy, dz, dyaw) x 8 + (position / 10, forward) per CAM_FRAME


def ego_vector(ego2world, k, chase_cfg=None, cams=None):
    """Rig motion of frames k+1..k+8 relative to frame k, plus the camera: either the fixed chase config
    (back, height; v0.2 chase models) or `cams` = c2rig [len(CAM_FRAMES), 4, 4] (free-view models). ego2world [N,4,4]."""
    import numpy as np
    inv = np.linalg.inv(ego2world[k])
    v = []
    for j in range(1, 9):
        r = inv @ ego2world[k + j]
        v += [r[0, 3], r[1, 3], r[2, 3], np.arctan2(r[1, 0], r[0, 0])]
    v = np.array(v, np.float32)
    v[0::4] /= 10.0; v[1::4] /= 2.0; v[2::4] /= 0.5; v[3::4] /= 0.3     # rough unit scaling
    if cams is None:
        return np.concatenate([v, [chase_cfg["back"] / 10.0, chase_cfg["height"] / 5.0]]).astype(np.float32)
    c = [np.r_[np.asarray(m)[:3, 3] / 10.0, np.asarray(m)[:3, 2]] for m in cams]
    return np.concatenate([v] + c).astype(np.float32)
