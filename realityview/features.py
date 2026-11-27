"""DINOv2-L patch tokens of the surround cameras (the generator's appearance context)."""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import paths

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_dinov2(device="cuda"):
    """Uses a local checkout ($RV_MODELS/dinov2 + dinov2_vitl14_pretrain.pth) if present, else torch.hub."""
    repo, weights = paths.model("dinov2"), paths.model("dinov2_vitl14_pretrain.pth")
    if os.path.isdir(repo) and os.path.exists(weights):
        net = torch.hub.load(repo, "dinov2_vitl14", source="local", pretrained=False)
        net.load_state_dict(torch.load(weights, map_location="cpu"))
    else:
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    return net.to(device).half().eval()


def load_dinov3(path=None, device="cuda"):
    """DINOv3 ViT-L/16 (Hugging Face transformers format, gated: facebook/dinov3-vitl16-pretrain-lvd1689m)."""
    from transformers import AutoModel
    # bf16, not fp16: DINOv3 activations overflow fp16 (every token comes out NaN)
    net = AutoModel.from_pretrained(path or paths.model("dinov3-vitl16"), dtype=torch.bfloat16)
    return net.to(device).eval()


@torch.no_grad()
def dino3_tokens(net, imgs):
    """Same layout as dino_tokens: [n,112,1024], 16x28 patch grid (448x256 at patch 16), 2x2 average pooled to 8x14."""
    dev = next(net.parameters()).device
    x = torch.from_numpy(np.stack([cv2.resize(i, (448, 256), interpolation=cv2.INTER_AREA) for i in imgs])).to(dev).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    h = net(pixel_values=((x - mean) / std).bfloat16()).last_hidden_state  # [n, 1 + registers + 448, 1024]
    t = h[:, -16 * 28:].reshape(-1, 16, 28, h.shape[-1]).permute(0, 3, 1, 2)
    return DINO3_SCALE * F.avg_pool2d(t.float(), 2).flatten(2).transpose(1, 2)


# DINOv3 tokens have ~1/4.8 the std of DINOv2's (0.25 vs 1.2 on the teacher frames); matched so a student
# initialised with DINOv2 feature projections starts at the right input scale
DINO3_SCALE = 4.8


@torch.no_grad()
def dino_tokens(net, imgs):
    """list of HxWx3 uint8 RGB -> [n,112,1024]: 16x28 patch grid at 392x224, 2x2 average pooled to 8x14."""
    dev = next(net.parameters()).device
    x = torch.from_numpy(np.stack([cv2.resize(i, (392, 224), interpolation=cv2.INTER_AREA) for i in imgs])).to(dev).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    t = net.forward_features(((x - mean) / std).half())["x_norm_patchtokens"].view(-1, 16, 28, 1024).permute(0, 3, 1, 2)
    return F.avg_pool2d(t.float(), 2).flatten(2).transpose(1, 2)
