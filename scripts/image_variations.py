from io import BytesIO
import os
from contextlib import nullcontext

import fire
import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
import requests

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def load_im(im_path):
    if im_path.startswith("http"):
        response = requests.get(im_path)
        response.raise_for_status()
        im = Image.open(BytesIO(response.content))
    else:
        im = Image.open(im_path).convert("RGB")
    tforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    inp = tforms(im).unsqueeze(0)
    return inp*2-1

@torch.no_grad()
def average_embed(embed1, embed2, weight: float = 0.5):
    """
    Average and normalize embeddings.
    """
    return F.normalize(
        embed1 * weight + embed2 * (1 - weight)
    )

@torch.no_grad()
def sample_model(input_im1, input_im2, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta):
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c1 = model.get_learned_conditioning(input_im1).tile(n_samples,1,1)
            c2 = model.get_learned_conditioning(input_im2).tile(n_samples,1,1)
            c = average_embed(c1, c2)

            if scale != 1.0:
                uc = torch.zeros_like(c)
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

def main(
    im1_path="data/textures/zebra.jpeg",
    im2_path="data/textures/pattern-small.jpeg",
    # im1_path="data/textures/franz-schekolin-IOiW0iGKwQg-unsplash.jpeg",
    # im2_path="data/textures/v-srinivasan-h64wUnq6ZxM-unsplash.jpeg",
    ckpt="models/ldm/stable-diffusion-v1/sd-clip-vit-l14-img-embed_ema_only.ckpt",
    config="configs/stable-diffusion/sd-image-condition-finetune.yaml",
    outpath="im_variations",
    scale=3.0,
    h=512,
    w=512,
    n_samples=4,
    precision="fp32",
    plms=True,
    ddim_steps=50,
    ddim_eta=1.0,
    device_idx=0,
    ):

    device = f"cuda:{device_idx}"
    input_im1 = load_im(im1_path).to(device)
    input_im2 = load_im(im2_path).to(device)
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)

    if plms:
        sampler = PLMSSampler(model)
        ddim_eta = 0.0
    else:
        sampler = DDIMSampler(model)

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    x_samples_ddim = sample_model(input_im1, input_im2, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta)
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(x_sample.astype(np.uint8)).save(
            os.path.join(sample_path, f"{base_count:05}.png"))
        base_count += 1

if __name__ == "__main__":
    fire.Fire(main)
