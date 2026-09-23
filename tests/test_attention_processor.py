import math

import torch

from cross_attention_interventions.attention import MechanisticCrossAttentionProcessor
from cross_attention_interventions.trace import TraceController, TraceRecorder


class DummyAttention:
    def __init__(self, dim=4, heads=2):
        self.heads = heads
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=False), torch.nn.Identity()])
        self.spatial_norm = None
        self.group_norm = None
        self.norm_cross = False
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        with torch.no_grad():
            for layer in (self.to_q, self.to_k, self.to_v, self.to_out[0]):
                layer.weight.copy_(torch.eye(dim))

    def head_to_batch_dim(self, x):
        b, n, d = x.shape
        dh = d // self.heads
        return x.view(b, n, self.heads, dh).permute(0, 2, 1, 3).reshape(b * self.heads, n, dh)

    def batch_to_head_dim(self, x):
        bh, n, dh = x.shape
        b = bh // self.heads
        return x.view(b, self.heads, n, dh).permute(0, 2, 1, 3).reshape(b, n, self.heads * dh)

    def get_attention_scores(self, q, k, mask=None):
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        if mask is not None:
            scores = scores + mask
        return scores.softmax(dim=-1)

    def prepare_attention_mask(self, mask, target_length, batch_size):
        return mask


def reference(attn, hidden, context):
    q = attn.head_to_batch_dim(attn.to_q(hidden))
    k = attn.head_to_batch_dim(attn.to_k(context))
    v = attn.head_to_batch_dim(attn.to_v(context))
    probs = attn.get_attention_scores(q, k)
    out = torch.bmm(probs, v)
    out = attn.batch_to_head_dim(out)
    return attn.to_out[0](out)


def test_processor_preserves_attention_output_and_records():
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10
    context = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4) / 10
    attn = DummyAttention()
    recorder = TraceRecorder(map_size=2)
    proc = MechanisticCrossAttentionProcessor("block.attn2.processor", TraceController(recorder=recorder))
    expected = reference(attn, hidden, context)
    actual = proc(attn, hidden, encoder_hidden_states=context)
    assert torch.allclose(actual, expected, atol=1e-6)
    strengths = recorder.head_strengths()["block.attn2.processor"]
    assert strengths.shape == (2, 5)
