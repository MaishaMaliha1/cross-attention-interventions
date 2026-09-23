from __future__ import annotations

import torch


def projected_value_norms(values: torch.Tensor, out_weight: torch.Tensor) -> torch.Tensor:
    """Return ||V[h,k] W_O[h]||_2 for each batch/head/token.

    values: [B, H, K, D_h]
    out_weight: [D_out, H * D_h]
    """
    if values.ndim != 4:
        raise ValueError("values must have shape [B,H,K,Dh]")
    batch, heads, tokens, head_dim = values.shape
    if out_weight.ndim != 2 or out_weight.shape[1] != heads * head_dim:
        raise ValueError("output projection shape is incompatible with values")
    chunks = out_weight.view(out_weight.shape[0], heads, head_dim).permute(1, 2, 0)
    projected = torch.einsum("bhkd,hdo->bhko", values, chunks)
    return torch.linalg.vector_norm(projected, ord=2, dim=-1)


def contribution_norms(
    attention: torch.Tensor,
    values: torch.Tensor,
    out_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute the norm of the projected token contribution Delta.

    attention: [B, H, Q, K]
    values: [B, H, K, D_h]
    result: [B, H, Q, K]
    """
    magnitudes = projected_value_norms(values, out_weight)
    return attention * magnitudes.unsqueeze(2)


def normalize_with_smoothing(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if eps <= 0:
        raise ValueError("eps must be positive")
    y = x.to(torch.float64).clamp_min(0) + eps
    return y / y.sum(dim=-1, keepdim=True).clamp_min(eps)


def kl_sensitivity(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(P || Q) with shared-support smoothing and normalization."""
    if p.shape != q.shape:
        raise ValueError("p and q must have identical shared support")
    pp = normalize_with_smoothing(p, eps)
    qq = normalize_with_smoothing(q, eps)
    return torch.sum(pp * (torch.log(pp) - torch.log(qq)), dim=-1)
