import torch

from cross_attention_interventions.tokenization import remove_word, shared_support_projection, words_with_spans


def test_remove_word():
    p = "a frog eating a banana"
    assert remove_word(p, 1) == "a eating a banana"


def test_word_spans():
    x = words_with_spans("red bird flies")
    assert [w.text for w in x] == ["red", "bird", "flies"]
    assert x[1].index == 1


def test_shared_support_leaves_removed_position_zero():
    baseline_ids = [1, 10, 20, 30, 2, 0]
    changed_ids = [1, 10, 30, 2, 0, 0]
    baseline_active = [1, 1, 1, 1, 1, 0]
    changed_active = [1, 1, 1, 1, 0, 0]
    changed = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]])
    got = shared_support_projection(
        baseline_ids, baseline_active, changed_ids, changed_active, changed
    )
    assert got.shape[-1] == len(baseline_ids)
    assert got[0, 2].item() == 0.0
    assert got[0, 3].item() == 3.0
