# Cross-Attention Interventions

Reference implementation for **Mechanistic Interpretability of Text-to-Image Diffusion Models via Cross-Attention Interventions** (Findings of ACL 2026).

The package implements the paper's causal, norm-based analysis of Stable Diffusion cross-attention. It measures projected token contributions, constructs token grounding maps, computes head-resolved intervention sensitivity, aggregates sensitivity by linguistic category, validates sensitive heads by targeted removal, and supports inference-time token strengthening through the projected value stream.

## Supported diffusion checkpoints

- Stable Diffusion v1.4 (`CompVis/stable-diffusion-v1-4`)
- Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
- Stable Diffusion 2.0-base (`stabilityai/stable-diffusion-2-base`)

Custom Stable Diffusion checkpoints with the same diffusers UNet cross-attention interface can also be supplied by repository ID or local path.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

A CUDA-enabled PyTorch installation is recommended for full experiments.

## Quick start

Generate a baseline image together with norm-based token grounding and head-resolved intervention sensitivity:

```bash
cai explain   --model sd15   --prompt "a frog eating a banana after peeling it"   --output outputs/frog
```

Strengthen a selected prompt word using the most sensitive cross-attention heads:

```bash
cai steer   --model sd14   --prompt "a red car and a blue bicycle"   --word bicycle   --output outputs/bicycle
```

Run attribution evaluation from a JSONL manifest containing prompts and segmentation masks:

```bash
cai evaluate-grounding   --model sd20   --manifest data/grounding.jsonl   --output outputs/grounding.json
```

Run prompt-adherence evaluation from a JSONL prompt manifest:

```bash
cai evaluate-adherence   --model sd14   --manifest data/prompts.jsonl   --output outputs/adherence.json
```

## Method implemented

For cross-attention head `h`, spatial location `q`, token `i`, and denoising step `t`, the implementation uses the projected token contribution

`Delta[h,q,i,t] = A[h,q,i,t] * (V[h,i,t] W_O[h])`.

The grounding signal is the norm of this projected contribution, averaged across heads, layers, and denoising steps after bringing spatial maps to a common latent grid. Head-level token strengths are obtained by averaging the same norm across spatial locations and denoising steps. Each head's token-strength vector is normalized to a distribution, and causal sensitivity is computed from the KL divergence between the baseline distribution and the corresponding token-removal intervention on shared support with smoothing.

Token strengthening scales the selected token's value vector only in the highest-sensitivity cross-attention heads during the early denoising phase, leaving model weights unchanged.

## Output layout

An explanation run writes:

```text
output/
├── baseline.png
├── interventions/
├── grounding/
├── head_sensitivity.json
├── pos_sensitivity.json
├── tokens.json
└── metadata.json
```

Grounding maps are saved both as numerical arrays and normalized PNG visualizations. Head sensitivity is stored by UNet cross-attention module and head index.

## Dataset manifests

Grounding manifest, one JSON object per line:

```json
{"prompt":"a dog","entities":[{"word":"dog","mask":"masks/0001_dog.png","class_id":18}]}
```

Prompt-adherence manifest:

```json
{"prompt":"a cat and a dog sitting on a sofa","objects":["cat","dog"]}
```

Paths in a grounding manifest are resolved relative to the manifest file.

## Reproducible paired interventions

Every baseline/intervention group reuses the exact same initial latent tensor and inference configuration. A latent tensor may also be supplied explicitly for repeatable reruns across machines.

## Tests

The unit tests cover projected-contribution norms, shared-support alignment, smoothed KL sensitivity, map aggregation, targeted head selection, and evaluation metrics without downloading diffusion checkpoints.

```bash
pytest -q
```

## Citation

```bibtex
@inproceedings{maliha-hougen-2026-mechanistic,
  title     = {Mechanistic Interpretability of Text-to-Image Diffusion Models via Cross-Attention Interventions},
  author    = {Maliha, Maisha and Hougen, Dean F.},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
  pages     = {25287--25299}
}
```

## License

MIT License. See [LICENSE](LICENSE).
