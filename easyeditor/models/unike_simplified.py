"""Explicit online-simplified UniKE adapters for MiniGPT-4 and LLaVA-OneVision.

Refactored to match BLIP2 UniKE patterns:
- Feature shift is a pure retrieval+mixing function (no trainable delta)
- Memory captured from frozen original MLP output (not expanded output)
- ASAM is parameter-scale-aware (matching BLIP2 _asam_loss)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import yaml
import os

from ..util.hparams import HyperParams
from .wise.utils import blip2_multimodal_tokenize, multimodal_tokenize
from .wise.utils import brackets_to_periods, parent_module


class ExpandedSwiGLUMLP(nn.Module):
    """Frozen SwiGLU MLP plus a trainable paired intrinsic neuron bank."""
    def __init__(self, original, add_neuron_num):
        super().__init__()
        if not all(hasattr(original, name) for name in ("gate_proj", "up_proj", "down_proj")):
            raise TypeError("Expected a SwiGLU module with gate_proj/up_proj/down_proj")
        self.original = original
        for parameter in self.original.parameters():
            parameter.requires_grad_(False)
        hidden = self.original.gate_proj.in_features
        output = self.original.down_proj.out_features
        count = int(add_neuron_num)
        self.extra_gate = nn.Linear(hidden, count, bias=False).float().to(original.gate_proj.weight.device)
        self.extra_up = nn.Linear(hidden, count, bias=False).float().to(original.up_proj.weight.device)
        self.extra_down = nn.Linear(count, output, bias=False).float().to(original.down_proj.weight.device)
        # Use kaiming init for extra_gate/extra_up (matching BLIP2's ExpandedLinearOutput)
        # and small non-zero init for extra_down so all params receive gradients from step 1
        nn.init.kaiming_uniform_(self.extra_gate.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.extra_up.weight, a=5**0.5)
        nn.init.normal_(self.extra_down.weight, std=1e-3)

    def forward(self, hidden_states):
        base = self.original.down_proj(
            F.silu(self.original.gate_proj(hidden_states)) * self.original.up_proj(hidden_states)
        )
        extra_input = hidden_states.float()
        extra = self.extra_down(F.silu(self.extra_gate(extra_input)) * self.extra_up(extra_input))
        return base + extra.to(dtype=base.dtype)


@dataclass
class UniKESimplifiedHyperParams(HyperParams):
    alg_name: str
    model_name: str
    name: str
    tokenizer_class: str
    tokenizer_name: str
    device: int
    dtype: torch.dtype
    inner_params: List[str]
    l_ike_layers: List[str]
    edit_lr: float
    n_iter: int
    locality_weight: float
    feature_shift_scale: float
    retrieval_top_k: int
    using_asam: bool
    asam_epsilon: float
    asam_weight: float
    qformer_checkpoint: str
    qformer_name_or_path: str
    state_dict_file: str
    pretrained_ckpt: str
    file_type: str
    coco_image: str
    rephrase_image: str
    exact_match: bool = False
    sequential_edit: bool = False
    model_parallel: bool = False
    use_chat_template: bool = True
    objective_optimization: str = "only_label"
    add_neuron_num: int = 1
    semantic_gate: bool = True
    adam_eps: float = 1e-4

    @classmethod
    def from_hparams(cls, path):
        with open(path, encoding="utf-8") as f:
            c = HyperParams.construct_float_from_scientific_notation(yaml.safe_load(f))
        dtype = c.get("dtype")
        if isinstance(dtype, str): c["dtype"] = getattr(torch, dtype.replace("torch.", ""))
        if c.get("alg_name") != "UniKE-Simplified": raise ValueError("wrong alg_name")
        return cls(**c)


def _module(model, dotted):
    obj = model
    for part in brackets_to_periods(dotted).split('.'):
        obj = getattr(obj, part)
    return obj


def _target_loss(out, labels=None):
    labels = labels if labels is not None else getattr(out, "labels", None)
    logits = out.logits
    if labels is None:
        raise RuntimeError("LLaVA simplified output did not retain labels")
    sl, sx = labels[:, 1:], logits[:, :-1]
    valid = sl.ne(-100)
    if not valid.any(): raise RuntimeError("no supervised labels")
    loss = CrossEntropyLoss(reduction="none")(sx.reshape(-1, sx.size(-1)), sl.reshape(-1)).view_as(sl)
    return (loss * valid).sum() / valid.sum()


class _OnlineMemoryShift(nn.Module):
    """Pure retrieval + mixing feature shift, matching BLIP2's _FeatureShift.

    No trainable parameters — the memory is frozen and the shift is a
    deterministic function of the input and the frozen memory bank.
    """
    def __init__(self, memory, scale=0.1, top_k=40, semantic_gate=True):
        super().__init__()
        self.register_buffer("memory", F.normalize(memory.float(), p=2, dim=-1), persistent=False)
        self.scale = float(scale)
        self.top_k = int(top_k)
        self.semantic_gate = bool(semantic_gate)

    def forward(self, _m, _i, output):
        if isinstance(output, tuple):
            return (self._shift(output[0]), *output[1:])
        return self._shift(output)

    def _shift(self, x):
        # All similarity/norm computations in float32 for numerical stability
        q = F.normalize(x.float(), p=2, dim=-1)
        mem = self.memory.to(q.device)
        scores = torch.matmul(q, mem.T)
        k = min(self.top_k, mem.size(0))
        weights, indices = torch.topk(scores, k=k, dim=-1)
        retrieved = (F.softmax(weights, dim=-1).unsqueeze(-1) * mem[indices]).sum(dim=-2)
        retrieved = retrieved.to(dtype=x.dtype)
        # Norm-preserving: scale retrieved to match query norm
        retrieved = retrieved * (x.norm(p=2, dim=-1, keepdim=True) / retrieved.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8))
        alpha = F.cosine_similarity(retrieved.float(), x.float(), dim=-1).clamp(min=0).unsqueeze(-1) if self.semantic_gate else 1.0
        blended = (1.0 - self.scale * alpha) * x.float() + self.scale * alpha * retrieved
        # Preserve norm after mixing
        return (blended * (x.float().norm(p=2, dim=-1, keepdim=True) / blended.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8))).to(x.dtype)


# Backward-compatible alias
_MemoryDelta = _OnlineMemoryShift


class UniKESimplifiedEditor(nn.Module):
    def __init__(self, model, hp, device):
        super().__init__()
        self.model, self.hp, self.device = model, hp, device
        self.target_modules = []
        self.original_modules = []
        self._install_intrinsic_modules()
        self.handles, self.shifts = [], []
        for p in model.parameters(): p.requires_grad_(False)
        for _, _, original in self.original_modules:
            for p in original.parameters(): p.requires_grad_(False)
        for module in self.target_modules:
            for name, parameter in module.named_parameters():
                if name.startswith("extra_"):
                    parameter.requires_grad_(True)

    def _install_intrinsic_modules(self):
        for name in self.hp.l_ike_layers:
            parent = _module(self.model, name.rsplit('.', 1)[0])
            child = name.rsplit('.', 1)[-1]
            original = getattr(parent, child)
            expanded = ExpandedSwiGLUMLP(original, self.hp.add_neuron_num).to(self.device)
            setattr(parent, child, expanded)
            self.original_modules.append((parent, child, original))
            self.target_modules.append(expanded)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _capture_memory(self, inputs):
        """Capture frozen original MLP output as per-case memory.

        Hooks on the original (frozen) down_proj ensure the memory is
        purely from the base path, not contaminated by the trainable extra path.
        """
        captures = [[] for _ in self.target_modules]
        # Hook on the frozen original down_proj output
        hs = [entry[2].down_proj.register_forward_hook(
            lambda _m, _i, o, c=c: c.append(o.detach())
        ) for entry, c in zip(self.original_modules, captures)]
        with torch.no_grad():
            self._forward(inputs)
        for h in hs:
            h.remove()
        for m, c in zip(self.target_modules, captures):
            if not c:
                raise RuntimeError("failed to capture online memory")
            x = c[0].reshape(-1, c[0].shape[-1])
            shift = _MemoryDelta(x, self.hp.feature_shift_scale, self.hp.retrieval_top_k, self.hp.semantic_gate).to(self.device)
            self.shifts.append(shift)
            self.handles.append(m.register_forward_hook(shift))

    def _forward(self, inputs):
        if self.hp.model_name in ("minigpt4", "blip2"):
            return self.model(inputs)
        if hasattr(inputs, "keys"):
            return self.model(**dict(inputs))
        return self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

    def edit(self, edit_inputs, loc_inputs):
        self._capture_memory(edit_inputs)
        # Only the intrinsic extra_* parameters are trainable (no delta in shift)
        intrinsic = [p for module in self.target_modules for name, p in module.named_parameters() if name.startswith("extra_")]
        params = intrinsic
        opt = torch.optim.Adam(params, lr=float(self.hp.edit_lr), eps=float(self.hp.adam_eps))
        diagnostics = os.environ.get("UNIKE_DIAGNOSTICS", "0") == "1"
        for step in range(int(self.hp.n_iter)):
            opt.zero_grad(set_to_none=True)
            edit_output = self._forward(edit_inputs)
            edit_labels = None if self.hp.model_name in ("minigpt4", "blip2") else edit_inputs["labels"]
            reliability = _target_loss(edit_output, edit_labels)
            with torch.no_grad():
                base = self._forward(loc_inputs).logits.detach()
            post = self._forward(loc_inputs).logits
            n = min(base.size(1), post.size(1))
            locality = F.kl_div(F.log_softmax(post[:, -n:], -1), F.softmax(base[:, -n:], -1), reduction="batchmean")
            loss = reliability + float(self.hp.locality_weight) * locality
            if not torch.isfinite(loss):
                raise FloatingPointError("UniKE simplified loss non-finite")
            before = [p.detach().float().clone() for p in params] if diagnostics else None

            if self.hp.using_asam:
                # Accumulate base loss gradients FIRST (matching BLIP2 pattern)
                loss.backward(retain_graph=True)
                # Compute ASAM perturbation direction
                grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)
                scaled = [p.detach().abs() * g.float() for p, g in zip(params, grads) if g is not None]
                norm = torch.sqrt(torch.stack([item.pow(2).sum() for item in scaled]).sum()).clamp_min(1e-12)
                perturbations = []
                with torch.no_grad():
                    for p, g in zip(params, grads):
                        if g is None:
                            perturbations.append(None)
                            continue
                        scale = p.detach().abs() + 1e-2
                        delta = float(self.hp.asam_epsilon) * scale * scale * g.float() / norm
                        if p.dtype != delta.dtype:
                            delta = delta.to(p.dtype)
                        p.add_(delta)
                        perturbations.append(delta)
                try:
                    asam_output = self._forward(edit_inputs)
                    asam = _target_loss(asam_output, edit_labels)
                    (float(self.hp.asam_weight) * asam).backward()  # ADDS to base gradients
                finally:
                    # Always restore parameters, even if forward/backward fails
                    with torch.no_grad():
                        for p, d in zip(params, perturbations):
                            if d is not None:
                                p.sub_(d.to(p.dtype))
                print(f"unike_asam_step={step} asam_loss={asam.detach()}")
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if diagnostics:
                assert before is not None
                grad_norms = [float(p.grad.detach().float().norm().item()) if p.grad is not None else 0.0 for p in params]
                update_norms = [float((p.detach().float() - b).norm().item()) for p, b in zip(params, before)]
                print(f"UNIKE_DIAGNOSTIC step={step} grad_norms={grad_norms} update_norms={update_norms}")
            print(f"unike_step={step} total_loss={loss.detach()}")

    def reset_layer(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.shifts.clear()
        for parent, child, original in self.original_modules:
            setattr(parent, child, original)


def apply_unike_simplified_to_multimodal_model(model, tok, requests, hparams, copy=False, **kwargs):
    if len(requests) != 1:
        raise ValueError("simplified UniKE supports singleton edits")
    req = requests[0]
    device = f"cuda:{hparams.device}"
    if hparams.model_name == "minigpt4":
        inputs, _, _, _, _ = blip2_multimodal_tokenize([req], tok, device, hparams=hparams)
    else:
        inputs, _, _, _, _ = multimodal_tokenize([req], tok, device, hparams=hparams)
    editor = UniKESimplifiedEditor(model, hparams, device)
    editor.edit(inputs[0], inputs[1])
    return editor, editor.reset_layer