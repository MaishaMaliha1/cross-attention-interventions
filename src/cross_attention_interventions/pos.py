from __future__ import annotations

from collections import defaultdict

from .tokenization import words_with_spans


def tag_words(prompt: str, model_name: str = "vblagoje/bert-english-uncased-finetuned-pos"):
    """Return coarse linguistic categories for prompt words.

    A transformers token-classification model is used when available. The result is
    aligned back to whitespace-delimited prompt words.
    """
    from transformers import pipeline

    tagger = pipeline("token-classification", model=model_name, aggregation_strategy="simple")
    words = words_with_spans(prompt)
    pred = tagger(prompt)
    groups = []
    for w in words:
        overlapping = [p for p in pred if max(w.start, p["start"]) < min(w.end, p["end"])]
        label = overlapping[0]["entity_group"] if overlapping else "X"
        groups.append((w.index, w.text, coarse_pos(label)))
    return groups


def coarse_pos(label: str) -> str:
    x = label.upper()
    if x.startswith(("NN", "NOUN", "PROPN")):
        return "NOUN"
    if x.startswith(("JJ", "ADJ")):
        return "ADJ"
    if x.startswith(("VB", "VERB", "AUX")):
        return "VERB"
    if x.startswith(("RB", "ADV")):
        return "ADV"
    if x in {"IN", "ADP"}:
        return "ADP"
    if x in {"DT", "DET"}:
        return "DET"
    if x in {"CC", "CCONJ", "SCONJ", "CONJ"}:
        return "CONJ"
    return "FUNC"


def aggregate_pos_sensitivity(tagged_words, word_sensitivity):
    acc = defaultdict(lambda: defaultdict(list))
    for word_index, _text, pos in tagged_words:
        if word_index not in word_sensitivity:
            continue
        for layer, vals in word_sensitivity[word_index].items():
            for h, value in enumerate(vals):
                acc[pos][(layer, h)].append(float(value))
    out = {}
    for pos, heads in acc.items():
        out[pos] = {
            f"{layer}::h{h}": sum(values) / len(values)
            for (layer, h), values in heads.items()
        }
    return out
