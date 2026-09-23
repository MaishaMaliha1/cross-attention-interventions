from __future__ import annotations

import torch

from .math_utils import contribution_norms
from .trace import HeadTarget, TraceController


class MechanisticCrossAttentionProcessor:
    """Diffusers attention processor that preserves attention behavior while tracing contributions."""

    def __init__(self, name: str, controller: TraceController):
        self.name = name
        self.controller = controller

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        spatial_hw = None

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            spatial_hw = (height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if getattr(attn, "norm_cross", False):
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        batch_size = hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, encoder_hidden_states.shape[1], batch_size
            )

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            query = attn.to_q(hidden_states)

        heads = attn.heads
        inner_dim = value.shape[-1]
        head_dim = inner_dim // heads
        values_bhkd = value.view(batch_size, -1, heads, head_dim).permute(0, 2, 1, 3).contiguous()

        plan = self.controller.steering
        if plan is not None and self.controller.step_index < plan.active_steps:
            values_bhkd = values_bhkd.clone()
            target_heads = [t.head for t in plan.heads if t.module == self.name]
            if target_heads:
                rows = range(batch_size // 2, batch_size) if batch_size > 1 else range(batch_size)
                for b in rows:
                    for h in target_heads:
                        if 0 <= h < heads:
                            valid = [i for i in plan.token_indices if i < values_bhkd.shape[2]]
                            if valid:
                                values_bhkd[b, h, valid] *= plan.factor
            value = values_bhkd.permute(0, 2, 1, 3).reshape(batch_size, -1, inner_dim)

        query_b = attn.head_to_batch_dim(query)
        key_b = attn.head_to_batch_dim(key)
        value_b = attn.head_to_batch_dim(value)
        probs_b = attn.get_attention_scores(query_b, key_b, attention_mask)
        hidden_states = torch.bmm(probs_b, value_b)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.controller.recorder is not None:
            q_len = probs_b.shape[1]
            k_len = probs_b.shape[2]
            probs = probs_b.view(batch_size, heads, q_len, k_len)
            weight = attn.to_out[0].weight
            delta = contribution_norms(probs, values_bhkd, weight)
            self.controller.recorder.add(self.name, delta, spatial_hw)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


def install_cross_attention_processors(unet, controller: TraceController) -> dict[str, object]:
    original = dict(unet.attn_processors)
    replacements = {}
    for name, proc in original.items():
        if ".attn2.processor" in name:
            replacements[name] = MechanisticCrossAttentionProcessor(name, controller)
        else:
            replacements[name] = proc
    unet.set_attn_processor(replacements)
    return original
