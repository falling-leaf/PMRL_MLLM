"""Outer-loop ASAM on MEND hypernetwork parameters."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch

from ...util.pmrl_utils import apply_asam_deltas, asam_parameter_deltas


def _finite_grads(params: Sequence[torch.nn.Parameter]) -> bool:
    for parameter in params:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return any(parameter.grad is not None for parameter in params)


def apply_outer_asam_grads(
    params: Sequence[torch.nn.Parameter],
    first_grads: Sequence[Optional[torch.Tensor]],
    second_pass: Callable[[], None],
    epsilon: float,
    rho: float = 0.1,
    replace: bool = True,
    exclude: Optional[Sequence[torch.nn.Parameter]] = None,
) -> Dict[str, object]:
    """Perturb ``params``, run ``second_pass``, restore weights, leave SAM grads.

    ``first_grads`` are detached clones of the *clipped* first-pass gradients.
    ``second_pass`` must write ``.grad`` on ``params`` (typically a replay of the
    accumulate window).  A non-finite second pass restores ``first_grads`` and
    the original weights; it never raises into the training loop.

    ``exclude`` parameters still get a second-pass gradient, but they are left
    out of the ASAM neighborhood.  MEND ``edit_lrs`` (~1e-4) would otherwise
    inherit the UniKE zero-init floor ``rho=0.1`` and dominate ``||T g||``.
    """
    params = list(params)
    first_grads = list(first_grads)
    if len(params) != len(first_grads):
        raise ValueError("ASAM first_grads must align with params")

    exclude_ids = {id(parameter) for parameter in (exclude or ())}
    delta_grads = [
        None if id(parameter) in exclude_ids else gradient
        for parameter, gradient in zip(params, first_grads)
    ]
    deltas = asam_parameter_deltas(params, delta_grads, float(epsilon), rho=float(rho))
    snapshot = [parameter.detach().clone() for parameter in params]
    skipped = False
    apply_asam_deltas(params, deltas, sign=1.0)
    try:
        for parameter in params:
            parameter.grad = None
        second_pass()
        if not _finite_grads(params):
            skipped = True
    except Exception:
        skipped = True
    finally:
        with torch.no_grad():
            for parameter, original in zip(params, snapshot):
                parameter.copy_(original)

    if skipped:
        for parameter, gradient in zip(params, first_grads):
            parameter.grad = None if gradient is None else gradient.clone()
    elif not replace:
        for parameter, gradient in zip(params, first_grads):
            if gradient is None:
                continue
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                parameter.grad = parameter.grad + gradient
    return {"skipped": skipped}
