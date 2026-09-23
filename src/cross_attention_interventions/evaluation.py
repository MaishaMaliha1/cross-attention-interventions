from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image

from .analysis import trace_prompt, word_grounding_map
from .metrics import best_threshold_iou, mean, percentile_iou
from .models import make_initial_latents
from .tokenization import words_with_spans


def read_jsonl(path: str | Path):
    p = Path(path)
    for line in p.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _find_word_index(prompt: str, target: str) -> int:
    target = target.lower().strip()
    for w in words_with_spans(prompt):
        if w.text.lower().strip(".,;:!?\"\'") == target:
            return w.index
    raise ValueError(f"word {target!r} not found in prompt")


def evaluate_grounding(pipe, config, manifest: str | Path):
    base = Path(manifest).resolve().parent
    p80, pinf = [], []
    per_entity = []
    for item in read_jsonl(manifest):
        latents = make_initial_latents(pipe, config)
        trace = trace_prompt(pipe, item["prompt"], config, latents)
        for entity in item["entities"]:
            idx = _find_word_index(item["prompt"], entity["word"])
            heat = word_grounding_map(trace, item["prompt"], pipe.tokenizer, idx).numpy()
            mask = np.asarray(Image.open(base / entity["mask"]).convert("L")) > 127
            image = Image.fromarray((heat * 255).astype(np.uint8)).resize(
                (mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR
            )
            resized = np.asarray(image, dtype=np.float32) / 255.0
            a = percentile_iou(resized, mask, 80.0)
            b = best_threshold_iou(resized, mask)
            p80.append(a)
            pinf.append(b)
            per_entity.append({"word": entity["word"], "iou80": a, "iou_inf": b})
    return {"mIoU80": mean(p80), "mIoU_inf": mean(pinf), "entities": per_entity}


def _load_clip(device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model.to(device).eval(), preprocess, tokenizer


def clip_similarity(model, preprocess, tokenizer, image, text: str, device: str) -> float:
    import torch
    with torch.no_grad():
        im = preprocess(image).unsqueeze(0).to(device)
        tx = tokenizer([text]).to(device)
        a = model.encode_image(im)
        b = model.encode_text(tx)
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        return float((a * b).sum().item())


def evaluate_adherence(pipe, config, manifest: str | Path):
    model, preprocess, tokenizer = _load_clip(config.device)
    full, minimum_object = [], []
    rows = []
    for item in read_jsonl(manifest):
        latents = make_initial_latents(pipe, config)
        image = pipe(
            prompt=item["prompt"],
            num_inference_steps=config.profile.steps,
            guidance_scale=config.profile.guidance_scale,
            width=config.profile.width,
            height=config.profile.height,
            latents=latents,
        ).images[0]
        fs = clip_similarity(model, preprocess, tokenizer, image, item["prompt"], config.device)
        obj_scores = [
            clip_similarity(model, preprocess, tokenizer, image, obj, config.device)
            for obj in item.get("objects", [])
        ]
        mos = min(obj_scores) if obj_scores else fs
        full.append(fs)
        minimum_object.append(mos)
        rows.append({"prompt": item["prompt"], "full_similarity": fs, "minimum_object": mos})
    return {"full_prompt_similarity": mean(full), "minimum_object_similarity": mean(minimum_object), "items": rows}
