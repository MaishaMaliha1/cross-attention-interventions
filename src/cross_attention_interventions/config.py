from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class ModelProfile:
    alias: str
    repo_id: str
    steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    scheduler: Literal["ddim", "native"] = "ddim"


MODEL_PROFILES: dict[str, ModelProfile] = {
    "sd14": ModelProfile("sd14", "CompVis/stable-diffusion-v1-4", steps=30),
    "sd15": ModelProfile("sd15", "runwayml/stable-diffusion-v1-5", steps=30),
    "sd20": ModelProfile("sd20", "stabilityai/stable-diffusion-2-base", steps=30),
}


def resolve_model_profile(name_or_path: str) -> ModelProfile:
    if name_or_path in MODEL_PROFILES:
        return MODEL_PROFILES[name_or_path]
    return ModelProfile(alias="custom", repo_id=name_or_path)


@dataclass
class ExperimentConfig:
    model: str = "sd15"
    steps: int | None = None
    guidance_scale: float | None = None
    width: int | None = None
    height: int | None = None
    scheduler: Literal["ddim", "native"] | None = None
    smoothing: float = 1e-8
    map_size: int = 64
    top_heads: int = 5
    strengthen_factor: float = 1.25
    strengthen_steps: int = 25
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: str = "cuda"
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def profile(self) -> ModelProfile:
        base = resolve_model_profile(self.model)
        return ModelProfile(
            alias=base.alias,
            repo_id=base.repo_id,
            steps=self.steps if self.steps is not None else base.steps,
            guidance_scale=(
                self.guidance_scale if self.guidance_scale is not None else base.guidance_scale
            ),
            width=self.width if self.width is not None else base.width,
            height=self.height if self.height is not None else base.height,
            scheduler=self.scheduler if self.scheduler is not None else base.scheduler,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "steps": self.profile.steps,
            "guidance_scale": self.profile.guidance_scale,
            "width": self.profile.width,
            "height": self.profile.height,
            "scheduler": self.profile.scheduler,
            "smoothing": self.smoothing,
            "map_size": self.map_size,
            "top_heads": self.top_heads,
            "strengthen_factor": self.strengthen_factor,
            "strengthen_steps": self.strengthen_steps,
            "dtype": self.dtype,
            "device": self.device,
            **self.extra,
        }
