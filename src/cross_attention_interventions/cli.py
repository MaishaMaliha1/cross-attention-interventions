from __future__ import annotations

from pathlib import Path
import json

import torch
import typer
from rich import print

from .analysis import explain_prompt, save_explanation, steer_prompt
from .config import ExperimentConfig
from .evaluation import evaluate_adherence, evaluate_grounding
from .models import load_latents, load_pipeline
from .tokenization import words_with_spans

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _config(model: str, config: Path | None) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(config) if config else ExperimentConfig()
    cfg.model = model
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
        cfg.dtype = "float32"
    return cfg


def _load_optional_latents(pipe, cfg, latent_in: Path | None):
    if latent_in is None:
        return None
    dtype = next(pipe.unet.parameters()).dtype
    return load_latents(latent_in, cfg.device, dtype)


@app.command()
def explain(
    prompt: str = typer.Option(...),
    model: str = typer.Option("sd15"),
    output: Path = typer.Option(Path("outputs/explanation")),
    config: Path | None = typer.Option(None),
    latent_in: Path | None = typer.Option(None),
):
    """Generate causal token-removal explanations and norm-based grounding maps."""
    cfg = _config(model, config)
    pipe = load_pipeline(cfg)
    latents = _load_optional_latents(pipe, cfg, latent_in)
    baseline, interventions, latents = explain_prompt(pipe, prompt, cfg, latents)
    save_explanation(output, prompt, cfg, baseline, interventions, latents)
    print(f"[green]Wrote explanation to {output}[/green]")


@app.command()
def steer(
    prompt: str = typer.Option(...),
    word: str = typer.Option(...),
    model: str = typer.Option("sd14"),
    output: Path = typer.Option(Path("outputs/steering")),
    config: Path | None = typer.Option(None),
    latent_in: Path | None = typer.Option(None),
):
    """Strengthen a selected prompt word in the highest-sensitivity heads."""
    cfg = _config(model, config)
    pipe = load_pipeline(cfg)
    words = words_with_spans(prompt)
    choices = [w for w in words if w.text.lower().strip(".,;:!?\"\'") == word.lower()]
    if not choices:
        raise typer.BadParameter(f"word {word!r} not found in prompt")
    latents = _load_optional_latents(pipe, cfg, latent_in)
    baseline, steered, sensitivity, heads, latents = steer_prompt(
        pipe, prompt, choices[0].index, cfg, latents
    )
    output.mkdir(parents=True, exist_ok=True)
    baseline.save(output / "baseline.png")
    steered.save(output / "steered.png")
    torch.save({"latents": latents.detach().cpu()}, output / "initial_latents.pt")
    (output / "selected_heads.json").write_text(
        json.dumps([{"module": h.module, "head": h.head} for h in sorted(heads, key=lambda x: (x.module, x.head))], indent=2)
    )
    (output / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    print(f"[green]Wrote steering output to {output}[/green]")


@app.command("evaluate-grounding")
def evaluate_grounding_cmd(
    manifest: Path = typer.Option(...),
    model: str = typer.Option("sd20"),
    output: Path = typer.Option(Path("outputs/grounding.json")),
    config: Path | None = typer.Option(None),
):
    cfg = _config(model, config)
    pipe = load_pipeline(cfg)
    result = evaluate_grounding(pipe, cfg, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(result)


@app.command("evaluate-adherence")
def evaluate_adherence_cmd(
    manifest: Path = typer.Option(...),
    model: str = typer.Option("sd14"),
    output: Path = typer.Option(Path("outputs/adherence.json")),
    config: Path | None = typer.Option(None),
):
    cfg = _config(model, config)
    pipe = load_pipeline(cfg)
    result = evaluate_adherence(pipe, cfg, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    app()
