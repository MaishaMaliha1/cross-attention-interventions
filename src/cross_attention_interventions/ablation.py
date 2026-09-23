from __future__ import annotations

from dataclasses import dataclass

import torch

from .trace import HeadTarget, TraceController


@dataclass
class HeadMaskPlan:
    heads: set[HeadTarget]


class HeadMaskController(TraceController):
    def __init__(self, masked_heads: set[HeadTarget]):
        super().__init__()
        self.masked_heads = masked_heads


def install_head_mask(unet, masked_heads: set[HeadTarget]):
    """Install processors that zero selected cross-attention head outputs."""
    from .attention import MechanisticCrossAttentionProcessor

    class MaskingProcessor(MechanisticCrossAttentionProcessor):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
            # Replicate the standard path but apply a temporary output hook to selected head slices.
            original_to_out = attn.to_out[0]
            weight = original_to_out.weight
            bias = original_to_out.bias
            heads = attn.heads
            head_dim = weight.shape[1] // heads
            targets = [x.head for x in masked_heads if x.module == self.name]
            if not targets:
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
            masked_weight = weight.clone()
            for h in targets:
                if 0 <= h < heads:
                    masked_weight[:, h * head_dim : (h + 1) * head_dim] = 0
            class TempLinear(torch.nn.Module):
                def forward(self, x):
                    return torch.nn.functional.linear(x, masked_weight, bias)
            attn.to_out[0] = TempLinear()
            try:
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
            finally:
                attn.to_out[0] = original_to_out

    original = dict(unet.attn_processors)
    replacements = {}
    controller = TraceController()
    for name, proc in original.items():
        replacements[name] = MaskingProcessor(name, controller) if ".attn2.processor" in name else proc
    unet.set_attn_processor(replacements)
    return original
