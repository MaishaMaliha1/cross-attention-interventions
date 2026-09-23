from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F


@dataclass
class LayerAggregate:
    token_sum: torch.Tensor | None = None
    calls: int = 0


@dataclass
class TraceRecorder:
    map_size: int = 64
    maps_sum: torch.Tensor | None = None
    maps_count: int = 0
    layers: dict[str, LayerAggregate] = field(default_factory=lambda: defaultdict(LayerAggregate))

    def add(self, name: str, delta_norm: torch.Tensor, spatial_hw: tuple[int, int] | None) -> None:
        """Aggregate [B,H,Q,K] projected-contribution norms."""
        x = delta_norm.detach().to("cpu", dtype=torch.float32)
        if x.shape[0] > 1:
            x = x[-1:]  # conditional branch when classifier-free guidance is active
        token_by_head = x.mean(dim=2).squeeze(0)  # [H,K]
        layer = self.layers[name]
        layer.token_sum = token_by_head if layer.token_sum is None else layer.token_sum + token_by_head
        layer.calls += 1

        spatial = x.mean(dim=1).squeeze(0).transpose(0, 1)  # [K,Q]
        q = spatial.shape[-1]
        if spatial_hw is None:
            side = int(round(q ** 0.5))
            if side * side != q:
                return
            spatial_hw = (side, side)
        h, w = spatial_hw
        if h * w != q:
            return
        maps = spatial.reshape(spatial.shape[0], 1, h, w)
        maps = F.interpolate(
            maps,
            size=(self.map_size, self.map_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        self.maps_sum = maps if self.maps_sum is None else self.maps_sum + maps
        self.maps_count += 1

    def head_strengths(self) -> dict[str, torch.Tensor]:
        out = {}
        for name, layer in self.layers.items():
            if layer.token_sum is not None and layer.calls:
                out[name] = layer.token_sum / layer.calls
        return out

    def grounding_maps(self) -> torch.Tensor:
        if self.maps_sum is None or self.maps_count == 0:
            raise RuntimeError("no cross-attention maps were recorded")
        return self.maps_sum / self.maps_count


@dataclass(frozen=True)
class HeadTarget:
    module: str
    head: int


@dataclass
class SteeringPlan:
    token_indices: list[int]
    heads: set[HeadTarget]
    factor: float
    active_steps: int


@dataclass
class TraceController:
    recorder: TraceRecorder | None = None
    steering: SteeringPlan | None = None
    step_index: int = 0

    def reset(self) -> None:
        self.step_index = 0

    def on_step_end(self, _pipe, _step, _timestep, callback_kwargs):
        self.step_index += 1
        return callback_kwargs
