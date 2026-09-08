"""Isolated UniKE-style BLIP-2 editor.

Faithful portions of the released UniKE code:
* trainable paired FFN key/value expansion in each of the final four layers;
* a frozen per-case external hidden-state memory at those same layers;
* cosine-controlled, norm-preserving latent feature shifts.

The release contains only MiniGPT-4/Vicuna checkpoints, semantic encoder, latent
vector and per-case memories.  This module therefore supports official artifacts
when they are supplied in BLIP-2/OPT dimensions, and has an explicitly labelled
``online_simplified`` fallback that uses edit activations as a frozen per-case
memory.  It never reads UniKE MiniGPT-4 artifacts as BLIP-2 tensors.
"""

import copy as copy_module
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from ..wise.utils import brackets_to_periods, parent_module, blip2_multimodal_tokenize
from ..transformer_patcher.layers import ExpandedLinearInput, ExpandedLinearOutput


def _target_loss(outputs):
    labels, logits = outputs.labels, outputs.logits
    shift_labels, shift_logits = labels[:, 1:].contiguous(), logits[:, :-1].contiguous()
    valid = shift_labels.ne(-100)
    if not valid.any():
        raise RuntimeError("UniKE BLIP-2 edit has no supervised target tokens")
    values = CrossEntropyLoss(reduction="none")(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
    ).view_as(shift_labels)
    return (values * valid).sum() / valid.sum()


class _FeatureShift(torch.nn.Module):
    """Latent-IKE retrieval and dynamic semantic gate at one FFN output."""

    def __init__(self, memory: torch.Tensor, top_k: int, scale: float, semantic_gate: bool):
        super().__init__()
        if memory.ndim != 2:
            raise ValueError(f"UniKE memory must be [M,H], got {tuple(memory.shape)}")
        self.register_buffer("memory", F.normalize(memory.float(), p=2, dim=-1), persistent=False)
        self.top_k, self.scale, self.semantic_gate = int(top_k), float(scale), bool(semantic_gate)

    def forward(self, _module, _inputs, output):
        # OPT MLP returns a tensor.  Retain original dtype and do not mutate state.
        if not torch.is_tensor(output):
            raise TypeError("UniKE BLIP-2 expects tensor FFN outputs")
        x = output
        query = F.normalize(x.float(), p=2, dim=-1)
        scores = torch.matmul(query, self.memory.T)
        k = min(self.top_k, self.memory.size(0))
        weights, indices = torch.topk(scores, k=k, dim=-1)
        retrieved = self.memory[indices]
        retrieved = (F.softmax(weights, dim=-1).unsqueeze(-1) * retrieved).sum(dim=-2)
        retrieved = retrieved.to(dtype=x.dtype)
        retrieved = retrieved * (x.norm(p=2, dim=-1, keepdim=True) / retrieved.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8))
        alpha = F.cosine_similarity(retrieved.float(), x.float(), dim=-1).clamp(min=0).unsqueeze(-1) if self.semantic_gate else 1.0
        blended = (1.0 - self.scale * alpha) * x + self.scale * alpha * retrieved
        # The released UniKE adapter scales the retrieved vector before mixing;
        # normalize again after mixing so the shift cannot alter FFN magnitude.
        return blended * (x.norm(p=2, dim=-1, keepdim=True) / blended.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8))


class UniKEBLIP2(torch.nn.Module):
    def __init__(self, model, config, device):
        super().__init__()
        self.model, self.config, self._device = model, config, device
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.entries, self.handles, self._feature_shifts = [], [], []
        self._install_expansions()
        self.memory_mode = "official_artifact" if config.retrieved_knowledge_path else "online_simplified"

    def _resolve_pair(self, names):
        if len(names) != 2:
            raise ValueError("UniKE-BLIP2 inner_params must be fc1/fc2")
        names = [name[:-7] if name.endswith(".weight") else name for name in names]
        if not names[0].endswith(".fc1") or not names[1].endswith(".fc2"):
            raise ValueError("UniKE-BLIP2 needs OPT-style fc1/fc2 pairs")
        if names[0].rsplit(".", 1)[0] != names[1].rsplit(".", 1)[0]:
            raise ValueError("UniKE-BLIP2 fc1/fc2 must share an FFN")
        return names

    def _install_expansions(self):
        fc1_name, fc2_name = self._resolve_pair(self.config.inner_params)
        template = fc1_name.rsplit(".layers.", 1)[0] + ".layers.{}.fc1"
        template2 = fc2_name.rsplit(".layers.", 1)[0] + ".layers.{}.fc2"
        layers = []
        for mlp_name in self.config.l_ike_layers:
            if ".layers." not in mlp_name:
                raise ValueError(f"Invalid UniKE l_ike layer: {mlp_name}")
            layers.append(int(mlp_name.rsplit(".layers.", 1)[1].split(".", 1)[0]))
        for layer in layers:
            first, second = template.format(layer), template2.format(layer)
            p1, p2 = parent_module(self.model, brackets_to_periods(first)), parent_module(self.model, brackets_to_periods(second))
            c1, c2 = first.rsplit(".", 1)[-1], second.rsplit(".", 1)[-1]
            original1, original2 = getattr(p1, c1), getattr(p2, c2)
            expanded1 = ExpandedLinearOutput(original1, self.config.add_neuron_num).to(self._device)
            expanded2 = ExpandedLinearInput(original2, self.config.add_neuron_num).to(self._device)
            # UniKE's added key/value path must be non-disruptive at start.  A
            # small key initialization and zero value coupling retain the base
            # FFN exactly while still allowing values to learn on step one.
            torch.nn.init.normal_(expanded1.extra_output.weight, std=1e-3)
            if expanded1.extra_output.bias is not None:
                torch.nn.init.zeros_(expanded1.extra_output.bias)
            torch.nn.init.zeros_(expanded2.extra_input.weight)
            setattr(p1, c1, expanded1); setattr(p2, c2, expanded2)
            self.entries.append((p1, c1, original1, expanded1, p2, c2, original2, expanded2))

    def _load_official_memory(self, sample_id: int, hidden_size: int):
        path = Path(self.config.retrieved_knowledge_path) / f"ike_{sample_id}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing UniKE per-case memory: {path}")
        memory = torch.load(path, map_location=self._device)
        if isinstance(memory, dict):
            memory = next((value for value in memory.values() if torch.is_tensor(value)), None)
        if not torch.is_tensor(memory):
            raise TypeError(f"Unsupported UniKE memory object at {path}")
        if memory.ndim == 3:  # published format [M, L, H]
            if memory.size(-1) != hidden_size:
                raise ValueError("Official UniKE memory hidden size is incompatible with BLIP-2 OPT")
            return [memory[:, layer_index, :] for layer_index in range(memory.size(1))]
        if memory.ndim == 2 and memory.size(-1) == hidden_size:
            return [memory] * len(self.entries)
        raise ValueError(f"Unsupported UniKE memory shape {tuple(memory.shape)}")

    def _online_memory(self, edit_inputs):
        captures, handles = [[] for _ in self.entries], []
        for capture, entry in zip(captures, self.entries):
            handles.append(entry[7].register_forward_hook(lambda _m, _i, out, c=capture: c.append(out.detach())))
        with torch.no_grad():
            self.model(edit_inputs)
        for handle in handles: handle.remove()
        if any(len(capture) != 1 for capture in captures):
            raise RuntimeError("UniKE online memory capture failed")
        # Hidden outputs carry target-edit context and remain frozen for the case.
        return [capture[0].reshape(-1, capture[0].size(-1)) for capture in captures]

    def install_memory(self, edit_inputs, sample_id: int):
        hidden_size = self.entries[0][7].linear.out_features
        memories = self._load_official_memory(sample_id, hidden_size) if self.config.retrieved_knowledge_path else self._online_memory(edit_inputs)
        if len(memories) == 1: memories *= len(self.entries)
        if len(memories) != len(self.entries):
            raise ValueError(f"UniKE memory has {len(memories)} layer slices; expected {len(self.entries)}")
        self._feature_shifts = []
        for entry, memory in zip(self.entries, memories):
            shift = _FeatureShift(memory.to(self._device), self.config.retrieval_top_k, self.config.feature_shift_scale, self.config.semantic_gate)
            self._feature_shifts.append((entry[7], shift))
            self.handles.append(entry[7].register_forward_hook(shift))

    def reset_layer(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()
        for p1, c1, o1, _e1, p2, c2, o2, _e2 in self.entries:
            setattr(p1, c1, o1); setattr(p2, c2, o2)

    def forward(self, *args, **kwargs): return self.model(*args, **kwargs)

    def _asam_loss(self, edit_inputs, locality_inputs):
        """One ASAM perturb-and-evaluate term over the added UniKE path.

        The perturbation is applied only to trainable added key/value parameters;
        frozen base weights and retrieval memories are untouched.
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        grads = torch.autograd.grad(self._base_loss(edit_inputs, locality_inputs), trainable, create_graph=False, allow_unused=True)
        # ASAM: maximize in a parameter-scale-aware neighborhood.  The frozen
        # backbone is excluded; only added UniKE key/value parameters move.
        scaled = [parameter.detach().abs() * gradient.float() for parameter, gradient in zip(trainable, grads) if gradient is not None]
        norm = torch.sqrt(torch.stack([item.pow(2).sum() for item in scaled]).sum()).clamp_min(1e-12)
        epsilon = float(self.config.asam_epsilon)
        perturbations = []
        with torch.no_grad():
            for parameter, gradient in zip(trainable, grads):
                if gradient is None:
                    perturbations.append(None)
                    continue
                scale = parameter.detach().abs() + 1e-2
                delta = epsilon * scale * scale * gradient.float() / norm
                if parameter.dtype != delta.dtype:
                    delta = delta.to(parameter.dtype)
                parameter.add_(delta)
                perturbations.append(delta)
        try:
            return self._base_loss(edit_inputs, locality_inputs), perturbations
        except Exception:
            with torch.no_grad():
                for parameter, delta in zip(trainable, perturbations):
                    if delta is not None:
                        parameter.sub_(delta)
            raise

    def _base_loss(self, edit_inputs, locality_inputs):
        reliability = _target_loss(self.model(edit_inputs))
        with torch.no_grad():
            base = self.model(locality_inputs).logits.detach()
        post = self.model(locality_inputs).logits
        length = min(base.size(1), post.size(1))
        locality = F.kl_div(F.log_softmax(post[:, -length:], dim=-1), F.softmax(base[:, -length:], dim=-1), reduction="batchmean")
        return reliability + float(self.config.locality_weight) * locality

    def edit(self, inputs):
        edit_inputs, locality_inputs = inputs
        params = []
        for _p1, _c1, _o1, e1, _p2, _c2, _o2, e2 in self.entries:
            params += list(e1.extra_output.parameters()) + list(e2.extra_input.parameters())
        optimizer = torch.optim.Adam(
            params, lr=float(self.config.edit_lr), eps=float(getattr(self.config, "adam_eps", 1e-4))
        )
        for step in range(int(self.config.n_iter)):
            optimizer.zero_grad(set_to_none=True)
            base_loss = self._base_loss(edit_inputs, locality_inputs)
            if not torch.isfinite(base_loss):
                raise FloatingPointError("UniKE BLIP-2 base loss is non-finite")
            base_loss.backward()
            loss = base_loss.detach()
            if bool(getattr(self.config, "using_asam", False)):
                asam, perturbations = self._asam_loss(edit_inputs, locality_inputs)
                if not torch.isfinite(asam):
                    raise FloatingPointError("UniKE BLIP-2 ASAM loss is non-finite")
                weighted_asam = float(self.config.asam_weight) * asam
                weighted_asam.backward()
                with torch.no_grad():
                    for parameter, delta in zip([p for p in self.parameters() if p.requires_grad], perturbations):
                        if delta is not None:
                            parameter.sub_(delta)
                loss = loss + weighted_asam.detach()
                print(f"unike_asam_step={step} asam_loss={asam.detach()}")
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            print(f"unike_step={step} total_loss={loss}")

    def _reinstall_expanded_layers(self):
        for p1, c1, _o1, e1, p2, c2, _o2, e2 in self.entries:
            setattr(p1, c1, e1); setattr(p2, c2, e2)

    def _reinstall_hooks(self):
        # Recreate hooks after a temporary unshifted locality forward.
        if self.handles:
            return
        self.handles = [module.register_forward_hook(shift) for module, shift in self._feature_shifts]
        if not self.handles:
            raise RuntimeError("UniKE feature-shift hooks unexpectedly absent")


def apply_unike_blip2_to_multimodal_model(model, tok, requests: List[Dict], hparams, copy=False, **kwargs: Any) -> Tuple[torch.nn.Module, Any]:
    if len(requests) != 1: raise ValueError("UniKE-BLIP2 supports singleton edits")
    if copy: model = copy_module.deepcopy(model).to(f"cuda:{hparams.device}")
    editor = UniKEBLIP2(model, hparams, f"cuda:{hparams.device}")
    request = requests[0]
    inputs, _, _, _, _ = blip2_multimodal_tokenize([request], processor=tok, device=f"cuda:{hparams.device}", context_templates=None, hparams=hparams)
    editor.install_memory(inputs[0], int(kwargs.get("sample_id", 0)))
    print(f"Executing UniKE-BLIP2 ({editor.memory_mode}): [{request['prompt']}] -> [{request['target']}]")
    editor.edit(inputs)
    return editor, editor.reset_layer
