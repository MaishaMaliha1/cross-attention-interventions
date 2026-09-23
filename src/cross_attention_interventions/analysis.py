from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image

from .attention import install_cross_attention_processors
from .config import ExperimentConfig
from .math_utils import kl_sensitivity
from .models import make_initial_latents
from .tokenization import (
    encode_prompt,
    remove_word,
    shared_support_projection,
    token_indices_for_word,
    words_with_spans,
)
from .trace import HeadTarget, SteeringPlan, TraceController, TraceRecorder


@dataclass
class RunTrace:
    image: Image.Image
    encoded: dict[str, object]
    maps: torch.Tensor
    head_strengths: dict[str, torch.Tensor]


def _run(pipe, prompt: str, config: ExperimentConfig, latents: torch.Tensor, controller: TraceController):
    controller.reset()
    result = pipe(
        prompt=prompt,
        num_inference_steps=config.profile.steps,
        guidance_scale=config.profile.guidance_scale,
        width=config.profile.width,
        height=config.profile.height,
        latents=latents.clone(),
        callback_on_step_end=controller.on_step_end,
    )
    return result.images[0]


def trace_prompt(pipe, prompt: str, config: ExperimentConfig, latents: torch.Tensor) -> RunTrace:
    recorder = TraceRecorder(map_size=config.map_size)
    controller = TraceController(recorder=recorder)
    original = install_cross_attention_processors(pipe.unet, controller)
    try:
        image = _run(pipe, prompt, config, latents, controller)
    finally:
        pipe.unet.set_attn_processor(original)
    return RunTrace(
        image=image,
        encoded=encode_prompt(pipe.tokenizer, prompt),
        maps=recorder.grounding_maps(),
        head_strengths=recorder.head_strengths(),
    )


def _shared_layer_keys(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    return sorted(set(a).intersection(b))


def intervention_sensitivity(
    baseline: RunTrace,
    changed: RunTrace,
    smoothing: float,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for layer in _shared_layer_keys(baseline.head_strengths, changed.head_strengths):
        p = baseline.head_strengths[layer]
        q_raw = changed.head_strengths[layer]
        q = shared_support_projection(
            baseline.encoded["ids"],
            baseline.encoded["active"],
            changed.encoded["ids"],
            changed.encoded["active"],
            q_raw,
        )
        if q.shape[-1] != p.shape[-1]:
            q = q[..., : p.shape[-1]]
        values = kl_sensitivity(p, q, eps=smoothing)
        result[layer] = values.to(torch.float64).tolist()
    return result


def explain_prompt(pipe, prompt: str, config: ExperimentConfig, latents: torch.Tensor | None = None):
    if latents is None:
        latents = make_initial_latents(pipe, config)
    baseline = trace_prompt(pipe, prompt, config, latents)
    words = words_with_spans(prompt)
    interventions = []
    all_sensitivity = []
    for word in words:
        changed_prompt = remove_word(prompt, word.index)
        changed = trace_prompt(pipe, changed_prompt, config, latents)
        sens = intervention_sensitivity(baseline, changed, config.smoothing)
        interventions.append((word, changed_prompt, changed, sens))
        all_sensitivity.append(sens)
    return baseline, interventions, latents


def word_grounding_map(baseline: RunTrace, prompt: str, tokenizer, word_index: int) -> torch.Tensor:
    words = words_with_spans(prompt)
    idx = token_indices_for_word(baseline.encoded, words[word_index])
    if not idx:
        raise ValueError(f"no tokenizer pieces found for word {words[word_index].text!r}")
    return baseline.maps[idx].mean(dim=0)


def rank_heads_for_word(sensitivity: dict[str, list[float]], top_k: int) -> list[HeadTarget]:
    flat = []
    for module, vals in sensitivity.items():
        flat.extend((float(v), HeadTarget(module, h)) for h, v in enumerate(vals))
    flat.sort(key=lambda x: x[0], reverse=True)
    return [target for _, target in flat[:top_k]]


def steer_prompt(
    pipe,
    prompt: str,
    word_index: int,
    config: ExperimentConfig,
    latents: torch.Tensor | None = None,
):
    if latents is None:
        latents = make_initial_latents(pipe, config)
    baseline = trace_prompt(pipe, prompt, config, latents)
    changed_prompt = remove_word(prompt, word_index)
    changed = trace_prompt(pipe, changed_prompt, config, latents)
    sensitivity = intervention_sensitivity(baseline, changed, config.smoothing)
    heads = set(rank_heads_for_word(sensitivity, config.top_heads))
    words = words_with_spans(prompt)
    token_idx = token_indices_for_word(baseline.encoded, words[word_index])
    plan = SteeringPlan(
        token_indices=token_idx,
        heads=heads,
        factor=config.strengthen_factor,
        active_steps=config.strengthen_steps,
    )
    controller = TraceController(steering=plan)
    original = install_cross_attention_processors(pipe.unet, controller)
    try:
        image = _run(pipe, prompt, config, latents, controller)
    finally:
        pipe.unet.set_attn_processor(original)
    return baseline.image, image, sensitivity, heads, latents


def normalize_map(x: torch.Tensor) -> np.ndarray:
    a = x.detach().cpu().numpy().astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def save_heatmap_png(x: torch.Tensor, path: Path) -> None:
    a = (normalize_map(x) * 255).astype(np.uint8)
    Image.fromarray(a, mode="L").resize((512, 512), Image.Resampling.BILINEAR).save(path)


def save_explanation(
    output: str | Path,
    prompt: str,
    config: ExperimentConfig,
    baseline: RunTrace,
    interventions,
    latents: torch.Tensor,
):
    out = Path(output)
    (out / "interventions").mkdir(parents=True, exist_ok=True)
    (out / "grounding").mkdir(parents=True, exist_ok=True)
    baseline.image.save(out / "baseline.png")
    torch.save({"latents": latents.detach().cpu()}, out / "initial_latents.pt")

    words = words_with_spans(prompt)
    for word in words:
        try:
            m = word_grounding_map(baseline, prompt, None, word.index)
        except ValueError:
            continue
        np.save(out / "grounding" / f"{word.index:02d}_{word.text}.npy", m.numpy())
        save_heatmap_png(m, out / "grounding" / f"{word.index:02d}_{word.text}.png")

    sensitivity_json = {}
    for word, changed_prompt, changed, sens in interventions:
        changed.image.save(out / "interventions" / f"{word.index:02d}_{word.text}.png")
        sensitivity_json[str(word.index)] = {
            "word": word.text,
            "prompt": changed_prompt,
            "layers": sens,
        }
    (out / "head_sensitivity.json").write_text(json.dumps(sensitivity_json, indent=2))
    (out / "tokens.json").write_text(json.dumps(baseline.encoded, indent=2))
    (out / "metadata.json").write_text(
        json.dumps({"prompt": prompt, "configuration": config.to_dict()}, indent=2)
    )
