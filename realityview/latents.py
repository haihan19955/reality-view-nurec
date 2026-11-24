"""Stable Diffusion VAE (sd-vae-ft-mse) used as the latent space of the generator."""
import torch

VAE_ID = "stabilityai/sd-vae-ft-mse"
SCALE = 0.18215


def load_vae(device="cuda"):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(VAE_ID).to(device).half().eval()


@torch.no_grad()
def encode(vae, imgs):
    """[n,H,W,3] uint8 RGB tensor -> [n,4,H/8,W/8] latents."""
    return vae.encode(imgs.permute(0, 3, 1, 2).half() / 127.5 - 1).latent_dist.mean * SCALE


@torch.no_grad()
def decode(vae, z):
    """[n,4,h,w] latents -> [n,3,8h,8w] float RGB in 0..255."""
    x = vae.decode(z.half() / SCALE).sample
    return (x.float().clamp(-1, 1) + 1) * 127.5
