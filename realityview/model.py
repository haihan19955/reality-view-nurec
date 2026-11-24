"""Pose-conditioned latent DiT trained with MeanFlow (one-step generation).

Inputs: noisy target latent z_t [B,4,hl,wl]; times (r, t); target-camera Plücker rays (one per 2x2 latent patch, rig frame);
surround camera tokens [B,6,112,256]; optional geometry splat latent [B,4,hl,wl] + coverage mask [B,1,hl,wl].
Output: average velocity u(z_t, r, t) over [r, t].  One-step sample: x0 = noise - u(noise, r=0, t=1).
Attention is written with plain matmuls so torch.func.jvp (forward-mode AD) works.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def t_embed(t, dim=256):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t[:, None].float() * 1000 * f[None]
    return torch.cat([a.cos(), a.sin()], -1)


class Attn(nn.Module):
    def __init__(self, d, h, kv_dim=None):
        super().__init__()
        self.h = h
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(kv_dim or d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, ctx):
        B, N, D = x.shape
        h = self.h
        q = self.q(x).view(B, N, h, D // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, h, D // h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) * (D // h) ** -0.5
        with torch.autocast("cuda", enabled=False):   # keep scores/softmax in bf16 (halves JVP memory)
            p = a.softmax(-1)
        return self.o((p.to(v.dtype) @ v).transpose(1, 2).reshape(B, N, D))


class Resampler(nn.Module):
    """Compress 6x112 camera tokens into n learned query tokens (runs outside the JVP, uses fused attention)."""

    def __init__(self, d, n=192, layers=2, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.layers = nn.ModuleList([nn.ModuleDict({"n": nn.LayerNorm(d), "a": nn.MultiheadAttention(d, heads, batch_first=True),
                                                    "n2": nn.LayerNorm(d), "m": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))})
                                     for _ in range(layers)])

    def forward(self, kv):
        x = self.q.expand(kv.shape[0], -1, -1)
        for L in self.layers:
            x = x + L["a"](L["n"](x), kv, kv, need_weights=False)[0]
            x = x + L["m"](L["n2"](x))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.sa = Attn(d, h)
        self.n2 = nn.LayerNorm(d, eps=1e-6)
        self.ca = Attn(d, h)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, ctx):
        s1, g1, s2, g2, c1, c2 = self.ada(F.silu(c)).chunk(6, -1)
        h = modulate(self.n1(x), s1, c1)
        x = x + g1[:, None] * self.sa(h, h)
        x = x + self.ca(self.n2(x), ctx)
        return x + g2[:, None] * self.mlp(modulate(self.n3(x), s2, c2))


class MVDiT(nn.Module):
    """77M-parameter DiT over 2x2 latent patches, cross-attending to the surround-camera tokens."""

    def __init__(self, d=512, depth=12, heads=8, tok_dim=256, n_cam=6, n_tok=112, hl=32, wl=32):
        super().__init__()
        self.d = d
        self.hl, self.wl = hl, wl
        self.ph, self.pw = hl // 2, wl // 2
        self.x_in = nn.Linear(9 * 4, d)        # (z_t 4 + geo 4 + mask 1) x 2x2 patch
        self.ray_in = nn.Linear(6, d)
        self.pos = nn.Parameter(torch.randn(1, self.ph * self.pw, d) * 0.02)
        self.ctx_in = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, d))
        self.cam_emb = nn.Parameter(torch.randn(1, n_cam, 1, d) * 0.02)
        self.tok_emb = nn.Parameter(torch.randn(1, 1, n_tok, d) * 0.02)
        self.resample = Resampler(d)
        self.t_mlp = nn.Sequential(nn.Linear(512, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.out_n = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.out_ada.weight)
        nn.init.zeros_(self.out_ada.bias)
        self.out = nn.Linear(d, 16)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def context(self, tok):
        """[B,6,112,tok_dim] camera tokens -> [B,192,d] context."""
        return self.resample((self.ctx_in(tok) + self.cam_emb + self.tok_emb).flatten(1, 2))

    def forward(self, z, r, t, ctx, rays, geo, gmask):
        B = z.shape[0]
        ph, pw = self.ph, self.pw
        x = torch.cat([z, geo, gmask], 1)                                     # [B,9,hl,wl]
        x = x.view(B, 9, ph, 2, pw, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, ph * pw, 36)
        x = self.x_in(x) + self.ray_in(rays) + self.pos
        c = self.t_mlp(torch.cat([t_embed(t), t_embed(t - r)], -1))
        for b in self.blocks:
            x = b(x, c, ctx)
        sh, sc = self.out_ada(F.silu(c)).chunk(2, -1)
        x = self.out(modulate(self.out_n(x), sh, sc))
        return x.view(B, ph, pw, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, 4, self.hl, self.wl)


@torch.no_grad()
def sample(model, ctx, rays, geo, gmask, steps=1, noise=None):
    """MeanFlow sampler: z_r = z_t - (t - r) * u(z_t, r, t). steps=1 is the one-step generator."""
    B = ctx.shape[0]
    z = noise if noise is not None else torch.randn(B, 4, model.hl, model.wl, device=ctx.device)
    ts = torch.linspace(1, 0, steps + 1, device=ctx.device)
    for i in range(steps):
        t = ts[i].expand(B)
        r = ts[i + 1].expand(B)
        z = z - (t - r)[:, None, None, None] * model(z, r, t, ctx, rays, geo, gmask)
    return z


def load_checkpoint(path, device="cuda"):
    """-> (EMA model in eval mode, checkpoint dict)."""
    ck = torch.load(path, map_location=device)
    model = MVDiT(hl=ck.get("hl", 32), wl=ck.get("wl", 32)).to(device).eval()
    model.load_state_dict(ck["ema"])
    return model, ck
