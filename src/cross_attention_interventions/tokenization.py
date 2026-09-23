from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


@dataclass(frozen=True)
class WordSpan:
    index: int
    text: str
    start: int
    end: int


def words_with_spans(prompt: str) -> list[WordSpan]:
    return [
        WordSpan(i, m.group(0), m.start(), m.end())
        for i, m in enumerate(re.finditer(r"\S+", prompt))
    ]


def remove_word(prompt: str, word_index: int) -> str:
    spans = words_with_spans(prompt)

    if not 0 <= word_index < len(spans):
        raise IndexError("word index out of range")

    keep = [span.text for span in spans if span.index != word_index]
    return " ".join(keep)


def encode_prompt(tokenizer, prompt: str) -> dict[str, object]:
    """Tokenize a prompt while retaining token-to-word alignment.

    Fast tokenizers provide character offsets directly. Stable Diffusion
    commonly uses a CLIP tokenizer configuration for which character offsets
    may not be available. In that case, token positions are reconstructed from
    tokenized prompt prefixes so that word-level grounding remains available.
    """

    kwargs = {
        "padding": "max_length",
        "max_length": tokenizer.model_max_length,
        "truncation": True,
        "return_tensors": "pt",
    }

    try:
        enc = tokenizer(
            prompt,
            return_offsets_mapping=True,
            **kwargs,
        )

        offsets_tensor = enc.pop("offset_mapping")
        offsets = offsets_tensor[0].tolist()

    except (TypeError, NotImplementedError, ValueError):
        enc = tokenizer(prompt, **kwargs)

        ids = enc["input_ids"][0].tolist()
        offsets = [(0, 0)] * len(ids)

        words = words_with_spans(prompt)

        # Stable Diffusion CLIP tokenizers use a beginning-of-text token.
        # Content token positions therefore begin after the prefix special
        # token. Determine the prefix length from an empty encoding rather
        # than assuming a specific tokenizer implementation.
        empty_ids = tokenizer(
            "",
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        empty_no_special_ids = tokenizer(
            "",
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        special_prefix = max(
            0,
            len(empty_ids) - len(empty_no_special_ids) - 1,
        )

        # CLIP normally has one beginning-of-text token. Keep a conservative
        # fallback for compatible tokenizers whose empty encoding differs.
        if special_prefix == 0:
            special_prefix = 1

        for word in words:
            prefix_before = prompt[: word.start]
            prefix_through = prompt[: word.end]

            before_ids = tokenizer(
                prefix_before,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            through_ids = tokenizer(
                prefix_through,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            start_idx = special_prefix + len(before_ids)
            end_idx = special_prefix + len(through_ids)

            # Respect the sequence length after truncation/padding.
            start_idx = max(0, min(start_idx, len(offsets)))
            end_idx = max(start_idx, min(end_idx, len(offsets)))

            for token_idx in range(start_idx, end_idx):
                offsets[token_idx] = (word.start, word.end)

    ids = enc["input_ids"][0].tolist()

    mask = enc.get("attention_mask")

    if mask is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

        active = [
            1 if pad_token_id is None or token_id != pad_token_id else 0
            for token_id in ids
        ]
    else:
        active = mask[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(ids)

    return {
        "ids": ids,
        "tokens": tokens,
        "active": active,
        "offsets": offsets,
    }


def token_indices_for_word(
    encoded: dict[str, object],
    word: WordSpan,
) -> list[int]:
    """Return tokenizer positions overlapping a prompt word."""

    offsets = encoded["offsets"]
    found: list[int] = []

    for token_idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue

        overlaps = max(start, word.start) < min(end, word.end)

        if overlaps:
            found.append(token_idx)

    return found


def active_indices(encoded: dict[str, object]) -> list[int]:
    """Return active, non-padding token positions."""

    return [
        token_idx
        for token_idx, flag in enumerate(encoded["active"])
        if flag
    ]


def shared_support_projection(
    baseline_ids: list[int],
    baseline_active: list[int],
    changed_ids: list[int],
    changed_active: list[int],
    changed_values,
):
    """Project an intervened token vector onto baseline token positions.

    Exact token-ID matching blocks are used to align the shortened prompt with
    the original prompt. Positions corresponding to removed tokens remain zero.

    This provides the shared support required before comparing baseline and
    intervention token-contribution distributions.
    """

    import torch

    baseline_indices = [
        i
        for i, active in enumerate(baseline_active)
        if active
    ]

    changed_indices = [
        i
        for i, active in enumerate(changed_active)
        if active
    ]

    baseline_sequence = [
        baseline_ids[i]
        for i in baseline_indices
    ]

    changed_sequence = [
        changed_ids[i]
        for i in changed_indices
    ]

    output = torch.zeros(
        *changed_values.shape[:-1],
        len(baseline_ids),
        dtype=changed_values.dtype,
        device=changed_values.device,
    )

    matcher = SequenceMatcher(
        a=baseline_sequence,
        b=changed_sequence,
        autojunk=False,
    )

    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            baseline_position = baseline_indices[block.a + offset]
            changed_position = changed_indices[block.b + offset]

            output[..., baseline_position] = changed_values[
                ..., changed_position
            ]

    return output
