"""Shared LAP/PMRL and adaptive-SAM helpers for multimodal editors."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch


def asam_parameter_deltas(
    params: Sequence[torch.nn.Parameter],
    grads: Sequence[Optional[torch.Tensor]],
    epsilon: float,
    rho: float = 0.1,
) -> List[Optional[torch.Tensor]]:
    """ASAM neighborhood with a scale floor so zero-init adapters still move.

    Kwon et al. use ``T = diag(|w|)``. Newly inserted UniKE / T-Patcher
    neurons often start at 0, which collapses that neighborhood.  A floor
    ``max(|w|, rho)`` keeps the first-step perturbation on the same order as
    the requested ``epsilon`` without touching frozen backbone weights.
    """
    scales: List[Optional[torch.Tensor]] = []
    weighted: List[torch.Tensor] = []
    for parameter, gradient in zip(params, grads):
        if gradient is None:
            scales.append(None)
            continue
        scale = torch.maximum(
            parameter.detach().abs().float(),
            parameter.new_tensor(float(rho), dtype=torch.float32),
        )
        scales.append(scale)
        weighted.append(scale * gradient.float())
    if not weighted:
        return [None for _ in params]
    norm = torch.sqrt(torch.stack([item.pow(2).sum() for item in weighted]).sum()).clamp_min(1e-12)
    deltas: List[Optional[torch.Tensor]] = []
    for parameter, gradient, scale in zip(params, grads, scales):
        if gradient is None or scale is None:
            deltas.append(None)
            continue
        delta = (float(epsilon) * scale * scale * gradient.float() / norm).to(dtype=parameter.dtype)
        deltas.append(delta)
    return deltas


def apply_asam_deltas(params: Sequence[torch.nn.Parameter], deltas: Sequence[Optional[torch.Tensor]], sign: float = 1.0) -> None:
    with torch.no_grad():
        for parameter, delta in zip(params, deltas):
            if delta is not None:
                parameter.add_(delta, alpha=sign)


def clone_request_view(request: dict, prompt=None, image=None) -> dict:
    """Shallow-copy an edit request, optionally swapping the gen-T / gen-M view."""
    view = dict(request)
    if prompt is not None:
        view["prompt"] = prompt
    if image is not None:
        view["image"] = image
    return view


def request_gen_views(request: dict, text_weight: float, image_weight: float) -> list:
    """Reliability views used to supervise Gen-T / Gen-M during the edit."""
    views = []
    if float(text_weight) and request.get("rephrase_prompt"):
        views.append((clone_request_view(request, prompt=request["rephrase_prompt"]), float(text_weight)))
    if float(image_weight) and request.get("image_rephrase") is not None:
        views.append((clone_request_view(request, image=request["image_rephrase"]), float(image_weight)))
    return views


def _sequence_mask(view: torch.Tensor, token_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if token_mask is None:
        return None
    mask = token_mask
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.ndim == 3:
        mask = mask.any(dim=-1)
    if view.ndim == 3 and mask.ndim == 2 and mask.shape[:2] == view.shape[:2]:
        return mask
    if view.ndim == 2 and mask.ndim == 1 and mask.shape[0] == view.shape[0]:
        return mask
    if view.ndim == 2 and mask.ndim == 2 and mask.numel() == view.shape[0]:
        return mask.reshape(-1)
    raise ValueError(
        f"PMRL token mask shape {tuple(mask.shape)} does not match view {tuple(view.shape)}"
    )


def select_pmrl_token_views(
    views: Iterable[torch.Tensor],
    token_mask: Optional[torch.Tensor] = None,
    pool: bool = False,
) -> List[torch.Tensor]:
    """Restrict PMRL to selected tokens; optionally mean-pool each sequence.

    Full-sequence alignment under a visual perturbation forces unrelated text
    tokens to stay put and is a common source of Loc-T / Acc regressions.
    Visual-token pooling leaves the edit-target path free while still asking
    the visual prefix to be stable across LAP views.
    """
    selected: List[torch.Tensor] = []
    for view in views:
        mask = _sequence_mask(view, token_mask)
        if pool and view.ndim == 3:
            if mask is None:
                pooled = view.float().mean(dim=1)
            else:
                weights = mask.to(dtype=view.dtype).unsqueeze(-1)
                pooled = (view.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            selected.append(pooled)
            continue
        if mask is None:
            selected.append(view.reshape(-1, view.size(-1)))
            continue
        selected.append(view[mask])
        if selected[-1].numel() == 0:
            raise RuntimeError("PMRL token mask selected no activations")
    shapes = {tuple(item.shape) for item in selected}
    if len(shapes) != 1:
        raise ValueError(f"PMRL view shapes differ after token selection: {shapes}")
    return selected


def visual_token_mask_from_labels(
    visual_mask: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    using_image_embedding: bool,
    joint_perturbation: bool,
) -> torch.Tensor:
    """Choose the LAP perturbation / PMRL support without touching answer tokens."""
    if labels is None:
        if visual_mask is None:
            raise RuntimeError("LAP requires a visual mask or labels")
        return visual_mask.bool()
    answer_mask = labels.ne(-100)
    if joint_perturbation:
        if attention_mask is None:
            return ~answer_mask
        return (~answer_mask) & attention_mask.bool()
    if using_image_embedding and visual_mask is not None:
        return visual_mask.bool()
    if attention_mask is None:
        return ~answer_mask
    return (~answer_mask) & attention_mask.bool()
