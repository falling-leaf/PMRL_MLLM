"""Transformer-Patcher for singleton multimodal knowledge editing.

Based on Huang et al. (ICLR 2023): add paired neurons to the final Transformer
FFN (fc1 output / fc2 input), freeze the original network, and optimize only
those new neurons for the requested edit.  This implementation adapts that
structural mechanism to BLIP-2's multimodal wrapper and preserves the original
model exactly before the first optimization step.
"""

import copy as copy_module
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from ..wise.utils import brackets_to_periods, parent_module, blip2_multimodal_tokenize, multimodal_tokenize
from .layers import ExpandedLinearInput, ExpandedLinearOutput


def _target_loss(outputs):
    labels = outputs.labels
    logits = outputs.logits
    shift_labels = labels[:, 1:].contiguous()
    shift_logits = logits[:, :-1, :].contiguous()
    mask = shift_labels.ne(-100)
    if not mask.any():
        raise RuntimeError("Transformer-Patcher edit has no supervised target tokens")
    token_loss = CrossEntropyLoss(reduction="none")(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
    ).view_as(shift_labels)
    return (token_loss * mask).sum() / mask.sum()


class TransformerPatcherMultimodal(torch.nn.Module):
    """Temporarily installs a trainable paired expansion in one FFN block."""

    def __init__(self, model, config, device):
        super().__init__()
        self.model = model
        self.config = config
        self._device = device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        add_count = int(config.add_neuron_num)
        self._swiglu = len(config.inner_params) == 3
        if self._swiglu:
            gate_name, up_name, down_name = self._resolve_swiglu_triplet(config.inner_params)
            self.gate_parent = parent_module(model, brackets_to_periods(gate_name))
            self.up_parent = parent_module(model, brackets_to_periods(up_name))
            self.fc2_parent = parent_module(model, brackets_to_periods(down_name))
            self.gate_child = gate_name.rsplit(".", 1)[-1]
            self.up_child = up_name.rsplit(".", 1)[-1]
            self.fc2_child = down_name.rsplit(".", 1)[-1]
            self.original_gate = getattr(self.gate_parent, self.gate_child)
            self.original_up = getattr(self.up_parent, self.up_child)
            self.original_fc2 = getattr(self.fc2_parent, self.fc2_child)
            self.patched_gate = ExpandedLinearOutput(self.original_gate, add_count).to(device)
            self.patched_up = ExpandedLinearOutput(self.original_up, add_count).to(device)
            self.patched_fc2 = ExpandedLinearInput(self.original_fc2, add_count).to(device)
            setattr(self.gate_parent, self.gate_child, self.patched_gate)
            setattr(self.up_parent, self.up_child, self.patched_up)
            setattr(self.fc2_parent, self.fc2_child, self.patched_fc2)
        else:
            fc1_name, fc2_name = self._resolve_pair(config.inner_params)
            self.fc1_parent = parent_module(model, brackets_to_periods(fc1_name))
            self.fc2_parent = parent_module(model, brackets_to_periods(fc2_name))
            self.fc1_child = fc1_name.rsplit(".", 1)[-1]
            self.fc2_child = fc2_name.rsplit(".", 1)[-1]
            self.original_fc1 = getattr(self.fc1_parent, self.fc1_child)
            self.original_fc2 = getattr(self.fc2_parent, self.fc2_child)
            self.patched_fc1 = ExpandedLinearOutput(self.original_fc1, add_count).to(device)
            self.patched_fc2 = ExpandedLinearInput(self.original_fc2, add_count).to(device)
            setattr(self.fc1_parent, self.fc1_child, self.patched_fc1)
            setattr(self.fc2_parent, self.fc2_child, self.patched_fc2)

    @staticmethod
    def _resolve_pair(inner_params):
        if len(inner_params) != 2:
            raise ValueError("Transformer-Patcher needs exactly [fc1.weight, fc2.weight]")
        names = [name[:-7] if name.endswith(".weight") else name for name in inner_params]
        if not names[0].endswith(".fc1") or not names[1].endswith(".fc2"):
            raise ValueError("Transformer-Patcher supports paired OPT-style fc1/fc2 FFNs")
        if names[0].rsplit(".", 1)[0] != names[1].rsplit(".", 1)[0]:
            raise ValueError("Transformer-Patcher fc1/fc2 must belong to one FFN block")
        return names

    @staticmethod
    def _resolve_swiglu_triplet(inner_params):
        """Validate one LLaMA/Qwen-style gated MLP triplet for later expansion."""
        if len(inner_params) != 3:
            raise ValueError("Transformer-Patcher needs [gate_proj.weight, up_proj.weight, down_proj.weight]")
        names = [name[:-7] if name.endswith(".weight") else name for name in inner_params]
        expected = (".gate_proj", ".up_proj", ".down_proj")
        if tuple(name.endswith(suffix) for name, suffix in zip(names, expected)) != (True, True, True):
            raise ValueError("Transformer-Patcher requires gate_proj/up_proj/down_proj ordering")
        parents = [name.rsplit(".", 1)[0] for name in names]
        if len(set(parents)) != 1 or not parents[0].endswith(".mlp"):
            raise ValueError("Transformer-Patcher SwiGLU projections must belong to one MLP block")
        return names

    def forward(self, *args, **kwargs):
        """Expose the wrapped MLLM API expected by the shared evaluator."""
        return self.model(*args, **kwargs)

    @property
    def device(self):
        return self.model.device

    def _install_patched_layers(self):
        if self._swiglu:
            setattr(self.gate_parent, self.gate_child, self.patched_gate)
            setattr(self.up_parent, self.up_child, self.patched_up)
        else:
            setattr(self.fc1_parent, self.fc1_child, self.patched_fc1)
        setattr(self.fc2_parent, self.fc2_child, self.patched_fc2)

    def reset_layer(self):
        if self._swiglu:
            setattr(self.gate_parent, self.gate_child, self.original_gate)
            setattr(self.up_parent, self.up_child, self.original_up)
        else:
            setattr(self.fc1_parent, self.fc1_child, self.original_fc1)
        setattr(self.fc2_parent, self.fc2_child, self.original_fc2)


    def _is_legacy_wrapper(self):
        return self.config.model_name in ("blip2", "minigpt4")

    def _forward_inputs(self, inputs):
        if isinstance(inputs, Mapping) and not self._is_legacy_wrapper():
            return self.model(**inputs)
        return self.model(inputs)

    def _base_losses(self, edit_inputs, locality_inputs):
        outputs = self._forward_inputs(edit_inputs)
        if self._is_legacy_wrapper():
            reliability_loss = _target_loss(outputs)
        else:
            reliability_loss = self._hf_target_loss(outputs.logits, edit_inputs["labels"])
        # Distill the untouched text-locality distribution from its frozen base
        # behavior; it is independent of the IC test answer labels.
        with torch.no_grad():
            self.reset_layer()
            base_logits = self._forward_inputs(locality_inputs).logits.detach()
            self._install_patched_layers()
        patched_logits = self._forward_inputs(locality_inputs).logits
        length = min(base_logits.size(1), patched_logits.size(1))
        locality_loss = F.kl_div(
            F.log_softmax(patched_logits[:, -length:, :], dim=-1),
            F.softmax(base_logits[:, -length:, :], dim=-1),
            reduction="batchmean",
        )
        if self.config.model_name in ("blip2", "minigpt4"):
            inputs_embeds, attention_mask, targets = self.model.image_encoding(edit_inputs)
            return reliability_loss, locality_loss, inputs_embeds, attention_mask, targets
        return reliability_loss, locality_loss, None, None, None

    @staticmethod
    def _pmrl_loss(views, tau_alignment, tau_regularization, alignment_weight, regularization_weight):
        if len(views) < 2:
            raise ValueError("PMRL requires original features plus at least one LAP view")
        shapes = [tuple(view.shape) for view in views]
        if len(set(shapes)) != 1:
            raise ValueError(f"T-Patcher PMRL view shapes differ: {shapes}")
        stacked = torch.stack([view.float() for view in views], dim=1)
        if not torch.isfinite(stacked).all():
            raise FloatingPointError("T-Patcher PMRL features contain NaN/Inf")
        normalized = F.normalize(stacked, p=2, dim=-1, eps=1e-6)
        anchor = normalized[:, :1, :].detach()
        cosine = (anchor * normalized[:, 1:, :]).sum(dim=-1)
        alignment = ((1.0 - cosine) / float(tau_alignment)).mean()
        consensus = F.normalize(normalized.mean(dim=1), p=2, dim=-1, eps=1e-6)
        logits = torch.matmul(consensus, consensus.T) / float(tau_regularization)
        targets = torch.arange(consensus.size(0), device=consensus.device)
        regularization = F.cross_entropy(logits, targets)
        balanced_regularization = (
            regularization / regularization.detach().clamp_min(1e-8)
            * alignment.detach().clamp_min(1e-8)
        )
        total = float(alignment_weight) * alignment + float(regularization_weight) * balanced_regularization
        if not torch.isfinite(total):
            raise FloatingPointError("T-Patcher PMRL loss is non-finite")
        print("tpatch_pmrl_alignment: {}, tpatch_pmrl_regularization: {}".format(alignment, regularization))
        return total

    def _build_lap_perturbations(
        self, base_embeds, gradient, perturb_end, num_views, epsilon,
        random_start=False, pgd_steps=1, step_size=None,
    ):
        """Create bounded visual-prefix LAP views without filesystem effects."""
        mask = torch.zeros(
            base_embeds.shape[:2], device=base_embeds.device, dtype=torch.bool
        )
        mask[:, :int(perturb_end)] = True
        mask_f = mask.unsqueeze(-1).to(base_embeds.dtype)
        masked_gradient = gradient * mask_f
        norm = masked_gradient.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        direction = masked_gradient.float() / norm.view(-1, 1, 1)
        direction = direction.to(masked_gradient.dtype)
        step_size = float(epsilon if step_size is None else step_size)
        views = []
        for view_index in range(int(num_views)):
            if random_start:
                delta = torch.randn_like(base_embeds) * mask_f
                delta_norm = delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                radius = torch.rand(delta.size(0), device=delta.device) * float(epsilon)
                delta = delta / delta_norm.to(delta.dtype).view(-1, 1, 1)
                delta = delta * radius.to(delta.dtype).view(-1, 1, 1)
                for _ in range(max(1, int(pgd_steps))):
                    delta = (delta + direction * step_size) * mask_f
                    delta_norm = delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                    factor = (float(epsilon) / delta_norm).clamp(max=1.0)
                    delta = delta * factor.to(delta.dtype).view(-1, 1, 1)
            else:
                delta = direction * (
                    float(epsilon) * float(view_index + 1) / float(num_views)
                )
            views.append(base_embeds.detach() + delta.detach())
        return views

    def _compute_lap_pmrl_loss(self, inputs_embeds, attention_mask, targets):
        probe = inputs_embeds.detach().clone().requires_grad_(True)
        probe_outputs, _ = self.model.LLM_forward(probe, attention_mask, targets)
        probe_loss = _target_loss(SimpleNamespace(logits=probe_outputs.logits, labels=targets))
        gradient = torch.autograd.grad(probe_loss, probe, retain_graph=False)[0]
        perturb_end = min(32, inputs_embeds.size(1)) if self.config.using_image_embedding else inputs_embeds.size(1)
        direction = gradient[:, :perturb_end, :]
        direction = direction / direction.float().flatten(1).norm(dim=1).clamp_min(1e-8).to(direction.dtype).view(-1, 1, 1)
        views = []
        target_weight = float(getattr(self.config, "lar_target_loss_weight", 0.0))
        target_consistency_loss = probe_loss.new_zeros(())
        random_start = bool(getattr(self.config, "lar_random_start", False))
        pgd_steps = int(getattr(self.config, "lar_pgd_steps", 1))
        step_size = float(getattr(self.config, "lar_step_size", self.config.lap_epsilon))
        if random_start:
            candidate_views = [inputs_embeds] + self._build_lap_perturbations(
                inputs_embeds,
                gradient,
                perturb_end,
                num_views=int(self.config.num_rephrase),
                epsilon=float(self.config.lap_epsilon),
                random_start=True,
                pgd_steps=pgd_steps,
                step_size=step_size,
            )
        else:
            candidate_views = [inputs_embeds]
            for view_idx in range(1, int(self.config.num_rephrase) + 1):
                current = inputs_embeds.detach().clone()
                current[:, :perturb_end, :] += direction.detach() * float(self.config.lap_epsilon) * view_idx
                candidate_views.append(current)
        for view_index, current in enumerate(candidate_views):
            captured = []
            handle = self.patched_fc2.register_forward_hook(lambda _m, _i, out: captured.append(out))
            view_outputs, _ = self.model.LLM_forward(current, attention_mask, targets)
            handle.remove()
            if len(captured) != 1:
                raise RuntimeError("T-Patcher LAP view capture failed")
            views.append(captured[0].reshape(-1, captured[0].size(-1)))
            if target_weight and view_index > 0:
                target_consistency_loss = target_consistency_loss + _target_loss(
                    SimpleNamespace(logits=view_outputs.logits, labels=targets)
                )
        pmrl_loss = self._pmrl_loss(
            views,
            self.config.pmrl_tau_alignment,
            self.config.pmrl_tau_regularization,
            self.config.pmrl_alignment_weight,
            self.config.pmrl_regularization_weight,
        ) * float(self.config.pmrl_scale)
        target_weight = float(getattr(self.config, "lar_target_loss_weight", 0.0))
        if target_weight:
            target_consistency_loss = target_consistency_loss / max(1, len(candidate_views) - 1)
            print("tpatch_lar_variant_target_loss: {}".format(target_consistency_loss))
            pmrl_loss = pmrl_loss + target_weight * target_consistency_loss
        return pmrl_loss

    @staticmethod
    def _hf_target_loss(logits, labels, reduction="mean"):
        shift_labels = labels[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        valid = shift_labels.ne(-100)
        if not valid.any():
            raise RuntimeError("Transformer-Patcher HF batch has no supervised target tokens")
        return CrossEntropyLoss(reduction=reduction)(shift_logits[valid], shift_labels[valid])

    def _prepare_hf_lap_inputs(self, inputs):
        core = self.model.model
        input_ids = inputs["input_ids"]
        text_embeds = core.get_input_embeddings()(input_ids)
        model_name = self.config.model_name.lower()
        if "qwen2-vl" in model_name:
            image_features = core.get_image_features(
                inputs["pixel_values"], inputs.get("image_grid_thw")
            )
        elif "llava-onevision" in model_name:
            image_features = core.get_image_features(
                inputs["pixel_values"], inputs["image_sizes"],
                batch_num_images=inputs.get("batch_num_images"),
            )
        else:
            raise NotImplementedError(f"Unsupported HF model: {self.config.model_name}")
        image_features = torch.cat(image_features, dim=0).to(text_embeds.device, text_embeds.dtype)
        image_mask, _ = core.get_placeholder_mask(
            input_ids, inputs_embeds=text_embeds, image_features=image_features
        )
        fused = text_embeds.masked_scatter(image_mask, image_features)
        kwargs = {"attention_mask": inputs["attention_mask"], "use_cache": False, "return_dict": True}
        if "qwen2-vl" in model_name:
            position_ids, _ = core.get_rope_index(
                input_ids, inputs.get("image_grid_thw"), inputs.get("video_grid_thw"), inputs["attention_mask"]
            )
            kwargs["position_ids"] = position_ids
        return fused, image_mask.any(dim=-1), kwargs

    def _build_masked_lap_perturbations(
        self, base_embeds, gradient, mask, num_views, epsilon,
        random_start=False, pgd_steps=1, step_size=None,
    ):
        mask_f = mask.unsqueeze(-1).to(base_embeds.dtype)
        masked_gradient = gradient * mask_f
        norm = masked_gradient.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        direction = masked_gradient.float() / norm.view(-1, 1, 1)
        direction = direction.to(masked_gradient.dtype)
        step_size = float(epsilon if step_size is None else step_size)
        variants = []
        for view_index in range(int(num_views)):
            if random_start:
                delta = torch.randn_like(base_embeds) * mask_f
                delta_norm = delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                radius = torch.rand(delta.size(0), device=delta.device) * float(epsilon)
                delta = delta / delta_norm.to(delta.dtype).view(-1, 1, 1)
                delta = delta * radius.to(delta.dtype).view(-1, 1, 1)
                for _ in range(max(1, int(pgd_steps))):
                    delta = (delta + direction * step_size) * mask_f
                    delta_norm = delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                    delta = delta * (float(epsilon) / delta_norm).clamp(max=1.0).to(delta.dtype).view(-1, 1, 1)
            else:
                delta = direction * (float(epsilon) * float(view_index + 1) / float(num_views))
            variants.append(delta.detach())
        return variants

    def _compute_hf_lap_pmrl_loss(self, inputs):
        fused, visual_mask, forward_kwargs = self._prepare_hf_lap_inputs(inputs)
        probe = fused.detach().clone().requires_grad_(True)
        probe_outputs = self.model(inputs_embeds=probe, **forward_kwargs)
        probe_loss = self._hf_target_loss(probe_outputs.logits, inputs["labels"], reduction="sum")
        gradient = torch.autograd.grad(probe_loss, probe, retain_graph=False)[0]
        answer_mask = inputs["labels"].ne(-100)
        if getattr(self.config, "lar_joint_perturbation", False):
            perturb_mask = (~answer_mask) & inputs["attention_mask"].bool()
        elif getattr(self.config, "using_image_embedding", False):
            perturb_mask = visual_mask
        else:
            perturb_mask = (~answer_mask) & inputs["attention_mask"].bool()
        if not perturb_mask.any():
            raise RuntimeError("Transformer-Patcher HF LAP perturbation mask is empty")
        mask_f = perturb_mask.unsqueeze(-1).to(fused.dtype)
        epsilon = float(self.config.lap_epsilon)
        views = []
        target_losses = []
        for delta in self._build_masked_lap_perturbations(
            fused, gradient, perturb_mask, int(self.config.num_rephrase), epsilon,
            random_start=bool(getattr(self.config, "lar_random_start", False)),
            pgd_steps=int(getattr(self.config, "lar_pgd_steps", 1)),
            step_size=float(getattr(self.config, "lar_step_size", epsilon)),
        ):
            current = fused.detach() + delta
            captured = []
            handle = self.patched_fc2.register_forward_hook(lambda _m, _i, out: captured.append(out))
            outputs = self.model(inputs_embeds=current, **forward_kwargs)
            handle.remove()
            if len(captured) != 1:
                raise RuntimeError("Transformer-Patcher HF LAP capture failed")
            views.append(captured[0].reshape(-1, captured[0].size(-1)))
            target_losses.append(self._hf_target_loss(outputs.logits, inputs["labels"]))
        base_capture = []
        handle = self.patched_fc2.register_forward_hook(lambda _m, _i, out: base_capture.append(out))
        self.model(inputs_embeds=fused, **forward_kwargs)
        handle.remove()
        if len(base_capture) != 1:
            raise RuntimeError("Transformer-Patcher HF base capture failed")
        pmrl = self._pmrl_loss(
            [base_capture[0].reshape(-1, base_capture[0].size(-1))] + views,
            self.config.pmrl_tau_alignment, self.config.pmrl_tau_regularization,
            self.config.pmrl_alignment_weight, self.config.pmrl_regularization_weight,
        ) * float(self.config.pmrl_scale)
        target_weight = float(getattr(self.config, "lar_target_loss_weight", 0.0))
        if target_weight:
            target = torch.stack(target_losses).mean()
            print("tpatch_lar_variant_target_loss: {}".format(target))
            pmrl = pmrl + target_weight * target
        return pmrl

    def _loss_with_optional_pmrl(self, edit_inputs, locality_inputs):
        reliability_loss, locality_loss, inputs_embeds, attention_mask, targets = self._base_losses(
            edit_inputs, locality_inputs
        )
        pmrl_loss = reliability_loss.new_zeros(())
        if self.config.using_extra and self.config.using_lap and self.config.using_pmrl:
            if self.config.model_name in ("blip2", "minigpt4"):
                pmrl_loss = self._compute_lap_pmrl_loss(inputs_embeds, attention_mask, targets)
            else:
                pmrl_loss = self._compute_hf_lap_pmrl_loss(edit_inputs)
        return reliability_loss + float(self.config.locality_weight) * locality_loss + pmrl_loss

    def edit(self, multimodal_inputs, n_iter, edit_lr, locality_weight):
        edit_inputs, locality_inputs = multimodal_inputs
        parameters = list(self.patched_fc2.extra_input.parameters())
        if self._swiglu:
            parameters += list(self.patched_gate.extra_output.parameters())
            parameters += list(self.patched_up.extra_output.parameters())
        else:
            parameters += list(self.patched_fc1.extra_output.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(edit_lr),
            eps=float(getattr(self.config, "adam_eps", 1e-8)),
        )
        for step in range(int(n_iter)):
            optimizer.zero_grad(set_to_none=True)
            loss = self._loss_with_optional_pmrl(edit_inputs, locality_inputs)
            if not torch.isfinite(loss):
                raise FloatingPointError("Transformer-Patcher loss is non-finite")
            loss.backward()
            optimizer.step()
            print("tpatch_step={} total_loss={}".format(step, loss.detach()))


def apply_transformer_patcher_to_multimodal_model(
    model, tok, requests: List[Dict], hparams, copy: bool = False, **kwargs: Any
) -> Tuple[torch.nn.Module, Any]:
    if len(requests) != 1:
        raise ValueError("Transformer-Patcher multimodal implementation supports singleton edits")
    device = f"cuda:{hparams.device}"
    if copy:
        model = copy_module.deepcopy(model).to(device)
    editor = TransformerPatcherMultimodal(model, hparams, device)
    request = requests[0]
    if hparams.model_name in ("blip2", "minigpt4"):
        multimodal_inputs, _, _, _, _ = blip2_multimodal_tokenize(
            [request], processor=tok, device=device, context_templates=None, hparams=hparams
        )
    elif "qwen2-vl" in hparams.model_name.lower() or "llava-onevision" in hparams.model_name.lower():
        multimodal_inputs, _, _, _, _ = multimodal_tokenize(
            [request], processor=tok, device=device, context_templates=None, hparams=hparams
        )
    else:
        raise NotImplementedError(f"Transformer-Patcher does not support model {hparams.model_name}")
    print("Executing Transformer-Patcher: [{}] -> [{}]".format(request["prompt"], request["target"]))
    editor.edit(
        multimodal_inputs,
        n_iter=hparams.n_iter,
        edit_lr=hparams.edit_lr,
        locality_weight=hparams.locality_weight,
    )
    return editor, editor.reset_layer
