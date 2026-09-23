from __future__ import annotations

import torch

from .config import ExperimentConfig


def torch_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_pipeline(config: ExperimentConfig):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    profile = config.profile
    dtype = torch_dtype(config.dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        profile.repo_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if profile.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(config.device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def make_initial_latents(pipe, config: ExperimentConfig, batch_size: int = 1) -> torch.Tensor:
    profile = config.profile
    channels = pipe.unet.config.in_channels
    scale = pipe.vae_scale_factor
    shape = (
        batch_size,
        channels,
        profile.height // scale,
        profile.width // scale,
    )
    dtype = next(pipe.unet.parameters()).dtype
    return torch.randn(shape, device=config.device, dtype=dtype)


def load_latents(path, device: str, dtype) -> torch.Tensor:
    data = torch.load(path, map_location=device, weights_only=True)
    if isinstance(data, dict):
        data = data["latents"]
    return data.to(device=device, dtype=dtype)
