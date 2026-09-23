import torch

from cross_attention_interventions.analysis import rank_heads_for_word
from cross_attention_interventions.trace import TraceRecorder


def test_trace_aggregates_maps_and_heads():
    rec = TraceRecorder(map_size=4)
    delta = torch.ones(1, 2, 4, 3)
    rec.add("layer", delta, (2, 2))
    maps = rec.grounding_maps()
    heads = rec.head_strengths()["layer"]
    assert maps.shape == (3, 4, 4)
    assert heads.shape == (2, 3)
    assert torch.allclose(maps, torch.ones_like(maps))
    assert torch.allclose(heads, torch.ones_like(heads))


def test_rank_heads():
    sens = {"a": [0.2, 0.9], "b": [0.5, 0.1]}
    got = rank_heads_for_word(sens, 2)
    assert [(x.module, x.head) for x in got] == [("a", 1), ("b", 0)]
