import copy
import logging
import os
from collections import defaultdict

import higher
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import deque
from higher.patch import (
    _MonkeyPatchBase,
    _torch,
    _typing,
    _utils,
    buffer_sync,
    make_functional,
)
from .patch import monkeypatch as _make_functional

from . import local_nn
from . import profiling
from .editable_model import EditableModel
from .hooks import hook_model, clear_mend_state, assert_mend_state_consumed
from ..utils import _inner_params, _logits, parent_module
from ..losses import hf_target_loss
from ...util.pmrl_utils import select_pmrl_token_views

LOG = logging.getLogger(__name__)

# ``MEND_NORM_SEQUENTIAL=1`` restores the original per-row Python Welford loop
# used before the block-wise update.  It exists purely to reproduce old runs
# bit for bit; the two implementations compute the same statistics.
_LEGACY_NORM_LOOP = os.environ.get("MEND_NORM_SEQUENTIAL", "0").lower() not in (
    "",
    "0",
    "false",
    "no",
)


class _FunctionalModel(nn.Module):
    """Run a module with differentiable fast weights without copying them.

    ``higher.monkeypatch(..., in_place=True)`` registers updated non-leaf fast
    weights as ``nn.Parameter`` objects while functionalizing large HF models.
    That registration detaches them from the MEND transform graph.  PyTorch's
    functional call substitutes tensors only for the duration of ``forward``,
    preserving their grad_fn and leaving the base model untouched.
    """

    def __init__(self, model, fast_params):
        super().__init__()
        # Do not register the 7B base model or non-leaf fast tensors as this
        # temporary wrapper's parameters.
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_fast_params", dict(fast_params))
        known = {name for name, _ in model.named_parameters()}
        missing = [key for key in self._fast_params if key not in known]
        if missing:
            raise KeyError(
                "fast_params keys are not in the wrapped model; "
                f"functional_call(strict=False) would silently skip the edit: {missing}"
            )

    def forward(self, *args, **kwargs):
        return torch.func.functional_call(
            self._model, self._fast_params, args=args, kwargs=kwargs, strict=False
        )


def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)

    return new_m, new_s


def batch_update_counter(x, m, s, k0):
    """Block form of :func:`update_counter` for a stack of rows.

    ``x`` is ``[n, dim]``, ``m``/``s`` are the running mean / sum of squared
    deviations of the ``k0`` samples seen so far.  Uses the standard parallel
    combination identity of Welford's recursion::

        k1 = k0 + n
        m1 = m0 + (mean(x) - m0) * n / k1
        s1 = s0 + sum((x - mean(x))**2) + (mean(x) - m0)**2 * k0 * n / k1

    which yields exactly the same statistics and the same ``k`` as applying
    :func:`update_counter` row by row (only the floating point reduction order
    differs).  It replaces ~13 tensor ops *per row* with a handful of ops for
    the whole block, which is what makes meta-training steps fast: the row loop
    issued ~1e5 micro-kernels per transform on a ~9k-row activation.
    """
    n = x.shape[0]
    batch_mean = x.mean(dim=0)
    centered = x - batch_mean
    batch_s = (centered * centered).sum(dim=0)
    k1 = k0 + n
    delta = batch_mean - m
    new_m = m + delta * (n / k1)
    new_s = s + batch_s + delta * delta * ((k0 * n) / k1)
    return new_m, new_s


class GradientTransform(nn.Module):
    def __init__(self, x_dim: int, delta_dim: int, cfg, n_modes=None):
        super().__init__()

        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.cfg = cfg
        if cfg.combine and (cfg.one_sided or cfg.x_only or cfg.delta_only):
            raise ValueError("cfg.combine cannot be used with one-sided MEND variants")

        self.norm_init = False
        self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))

        MlpClass = getattr(local_nn, cfg.mlp_class)
        LOG.info(f"Building Gradient Transform with MLP class {MlpClass}")

        def delta_net():
            return MlpClass(
                delta_dim,
                delta_dim,
                delta_dim * 2,
                cfg.n_hidden,
                init=cfg.init,
                act=cfg.act,
                rank=cfg.rank,
                n_modes=n_modes,
            )

        def x_net():
            return MlpClass(
                x_dim,
                x_dim,
                x_dim * 2,
                cfg.n_hidden,
                init=cfg.init,
                act=cfg.act,
                rank=cfg.rank,
                n_modes=n_modes,
            )

        def combined_net():
            return MlpClass(
                delta_dim + x_dim,
                delta_dim + x_dim,
                (delta_dim + x_dim) * 2,
                cfg.n_hidden,
                init=cfg.init,
                act=cfg.act,
                rank=cfg.rank,
                n_modes=n_modes,
            )

        def ID():
            return lambda x, mode=None: x

        if cfg.combine:
            self.mlp = combined_net()
        elif cfg.one_sided:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.x_only:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.delta_only:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

    def _update_running_stats(self, u_, v_):
        """Fold the captured rows into the running normalisation statistics.

        Both branches below implement the same Welford recursion; the default
        branch evaluates it block-wise (see :func:`batch_update_counter`) while
        ``MEND_NORM_SEQUENTIAL=1`` keeps the original per-row Python loop for
        bit-for-bit reproduction of runs made before this optimisation.
        """
        if _LEGACY_NORM_LOOP:
            for idx in range(u_.shape[0]):
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(
                        u_[idx], self.u_mean, self.u_s, self.k
                    )
                    self.v_mean, self.v_s = update_counter(
                        v_[idx], self.v_mean, self.v_s, self.k
                    )
            return

        if u_.shape[0] == 0:
            return

        if not self.norm_init:
            # First-ever pair initialises the statistics exactly as the
            # sequential loop did: mean = x[0], s = 0, k = 1.
            self.u_mean = u_[0].clone().detach()
            self.v_mean = v_[0].clone().detach()
            self.u_s = torch.zeros_like(self.u_mean)
            self.v_s = torch.zeros_like(self.v_mean)
            self.k[:] = 1
            self.norm_init = True
            u_, v_ = u_[1:], v_[1:]

        n_new = u_.shape[0]
        if n_new == 0:
            return

        k_prev = self.k
        self.u_mean, self.u_s = batch_update_counter(
            u_, self.u_mean, self.u_s, k_prev
        )
        self.v_mean, self.v_s = batch_update_counter(
            v_, self.v_mean, self.v_s, k_prev
        )
        self.k[:] = k_prev + n_new

    def forward(self, u, v, param_idx=None):
        u, v = u.to(torch.float32), v.to(torch.float32)

        u_ = u.view(-1, u.shape[-1])
        v_ = v.view(-1, v.shape[-1])

        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(
            -1
        )  # Skip batch elements with zero grad
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]

        if self.training:
            with profiling.section("mend.norm_stats"):
                self._update_running_stats(u_, v_)

            if self.cfg.norm and self.k >= 2:
                self.u_std = (self.u_s / (self.k - 1)) ** 0.5
                self.v_std = (self.v_s / (self.k - 1)) ** 0.5

        if self.cfg.norm and self.k >= 2:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
        else:
            # A singleton multimodal edit can provide only one non-zero
            # activation/gradient pair.  Online normalization is undefined
            # until a second sample exists; use the raw pair for this first
            # update instead of aborting the run or feeding NaNs.
            u_input = u_
            v_input = v_

        if self.cfg.combine:
            output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)
            out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            return out1, out2
        else:
            return self.mlp1(u_input, mode=param_idx), self.mlp2(
                v_input, mode=param_idx
            )


class MEND(EditableModel):
    def get_shape(self, p):
        # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
        return (
            p.shape
            if isinstance(self.model, transformers.GPT2LMHeadModel)
            else (p.shape[1], p.shape[0])
        )

    def __init__(self, model, config, model_constructor, mend=None, edit_lrs=None):
        super().__init__(model, config, model_constructor)

        configured_device = self.config.device
        if isinstance(configured_device, int) or str(configured_device).isdigit():
            index = int(configured_device)
            self.config.device = torch.device(
                f"cuda:{index}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.config.device = torch.device(str(configured_device))
        # BaseTrainer resolves the public config too; keep this wrapper's copy
        # in sync because EditableModel deep-copies the config above.
        config.device = self.config.device

        inner_names = set(self.config.inner_params)
        for n, p in model.named_parameters():
            # Do not rely on parameter ordering: HF multimodal target names can
            # be interleaved with frozen vision/language parameters.  Enabling
            # gradients only for the explicit MEND targets reduces backward
            # memory and prevents unrelated parameters from accumulating state.
            if n not in inner_names and p.is_leaf:
                p.requires_grad = False

        if edit_lrs is None:
            edit_lrs = nn.Parameter(
                torch.tensor([config.edit_lr] * len(self.config.inner_params))
            )
        self.edit_lrs = edit_lrs

        if not hasattr(self.model, "handles"):
            hook_model(self.model, self.config.inner_params)
            LOG.info(f"Hooked {len(self.model.handles)//2} modules")

        if config.shared:
            shape_dict = defaultdict(list)
            for n, p in _inner_params(
                model.named_parameters(), self.config.inner_params
            ):
                shape_dict[self.get_shape(p)].append(n)
            self.shape_dict = shape_dict

        if mend is None:
            if not config.shared:
                self.mend = nn.ModuleDict(
                    {
                        n.replace(".", "#"): GradientTransform(
                            *self.get_shape(p), config
                        )
                        for (n, p) in _inner_params(
                            model.named_parameters(), self.config.inner_params
                        )
                    }
                )
            else:
                self.mend = nn.ModuleDict(
                    {
                        str(tuple(s)): GradientTransform(
                            *s, config, len(shape_dict[s])
                        )
                        for s in shape_dict.keys()
                    }
                )
            if self.config.model_parallel:
                self.mend.to(deque(self.model.parameters(), maxlen=1)[0].device)
            else:
                self.mend.to(self.config.device)
        else:
            self.mend = mend

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            prefix=prefix, keep_vars=keep_vars
        )  # Get default state dict
        model_keys = self.model.state_dict(
            prefix=prefix, keep_vars=keep_vars
        ).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"model.{k}"]
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, False)
        # We should only have missing keys for the model, and no unexpected keys
        assert (
            len([k for k in res.missing_keys if not k.startswith("model.")]) == 0
        ), "Should only have missing keys for model, got " + str(
            [k for k in res.missing_keys if not k.startswith("model.")]
        )
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def forward(self, *inputs, **kwargs):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower():
            outputs = self.model(*inputs, **kwargs)
        elif "llava-onevision" in self.config.model_name.lower() or "qwen2-vl" in self.config.model_name.lower():
            multimodal_inputs = dict(inputs[0]) if inputs else dict(kwargs)
            # Do not ask Transformers to materialize the full-vocabulary CE
            # tensor on auxiliary MEND forwards. Losses are computed by the
            # bounded helpers below using the batch labels.
            multimodal_inputs.pop("labels", None)
            outputs = self.model(**multimodal_inputs)
        elif 'gpt' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'llama' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'chatglm2' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'internlm' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'qwen' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'mistral' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        else:
            outputs = _logits(self.model(**kwargs))
        return outputs
    
    def outer_parameters(self):
        return list(self.mend.parameters()) + [self.edit_lrs]

    def _hf_last_decoder_layer(self):
        core = self.model.model
        if hasattr(core, "language_model") and hasattr(core.language_model, "layers"):
            return core.language_model.layers[-1]
        if hasattr(core, "layers"):
            return core.layers[-1]
        raise AttributeError("Could not locate the last language-model decoder layer")

    @staticmethod
    def _hf_target_loss(logits, labels, reduction="mean", attention_mask=None):
        """Compatibility wrapper around the shared masked HF causal loss."""
        return hf_target_loss(
            logits, labels, attention_mask=attention_mask, reduction=reduction
        )

    def _prepare_hf_visual_mask(self, batch):
        core = self.model.model
        input_ids, labels = batch["input_ids"], batch["labels"]
        text_embeds = core.get_input_embeddings()(input_ids)
        if "llava-onevision" in self.config.model_name.lower():
            image_features = core.get_image_features(
                batch["pixel_values"], batch["image_sizes"],
                batch_num_images=batch.get("batch_num_images"),
            )
            if isinstance(image_features, (tuple, list)):
                image_features = torch.cat(image_features, dim=0)
        else:
            image_features = torch.cat(core.get_image_features(
                batch["pixel_values"], batch.get("image_grid_thw")
            ), dim=0).to(text_embeds.device, text_embeds.dtype)
        image_mask, _ = core.get_placeholder_mask(
            input_ids, inputs_embeds=text_embeds, image_features=image_features
        )
        perturb_mask = image_mask.any(dim=-1)
        if not getattr(self.config, "using_image_embedding", True):
            perturb_mask = (~labels.ne(-100)) & batch["attention_mask"].bool()
        if not perturb_mask.any():
            raise RuntimeError("MEND LAP perturbation region is empty")
        return perturb_mask

    def _visual_pmrl_from_views(self, live_hidden, views):
        selected = select_pmrl_token_views([live_hidden] + views, token_mask=None, pool=False)
        if len(selected) < 2:
            raise RuntimeError("MEND PMRL needs the original view plus at least one LAP view")
        flattened = [F.normalize(v.float(), dim=-1) for v in selected]
        live = flattened[0]
        alignment = torch.stack([1 - (live * v.detach()).sum(-1).mean() for v in flattened[1:]]).mean()
        alignment_loss = alignment / float(self.config.pmrl_tau_alignment)
        consensus = F.normalize(torch.stack(flattened, dim=1).mean(dim=1), dim=-1)
        logits = consensus @ consensus.T / float(self.config.pmrl_tau_regularization)
        target = torch.arange(logits.size(0), device=logits.device)
        regularization = F.cross_entropy(logits, target)
        balanced_regularization = (
            regularization
            / regularization.detach().clamp_min(1e-8)
            * alignment_loss.detach().clamp_min(1e-8)
        )
        pmrl = (
            float(self.config.pmrl_alignment_weight) * alignment_loss
            + float(self.config.pmrl_regularization_weight) * balanced_regularization
        ) * float(self.config.pmrl_scale)
        if not torch.isfinite(pmrl):
            raise FloatingPointError("MEND PMRL loss is non-finite")
        return pmrl

    def _hidden_space_visual_pmrl(self, live_visual):
        """UniKE-style neighborhood: perturb visual tokens in hidden space.

        Extra 7B LAP forwards OOM even at 83–95GB because they overlap the
        main MEND edit graph.  A detached hidden-space view keeps visual
        alignment in the inner loss without a second model pass.
        """
        epsilon = float(self.config.lap_epsilon)
        noise = torch.randn_like(live_visual)
        noise = noise / noise.float().norm(dim=-1, keepdim=True).clamp_min(1e-8).to(noise.dtype)
        view = (live_visual.detach() + epsilon * noise).detach()
        return self._visual_pmrl_from_views(live_visual, [view])

    def _maybe_mend_extra_loss(self, batch, base_hidden=None, training=True):
        if not (
            getattr(self.config, "using_extra", False)
            and getattr(self.config, "using_lap", False)
            and getattr(self.config, "using_pmrl", False)
        ):
            if base_hidden is not None:
                return base_hidden.new_zeros(())
            device = next(self.model.parameters()).device
            return torch.zeros((), device=device)
        if (
            "qwen2-vl" not in self.config.model_name.lower()
            and "llava-onevision" not in self.config.model_name.lower()
        ):
            raise NotImplementedError(
                "MEND LAP+PMRL currently supports Qwen2-VL and LLaVA-OneVision"
            )
        views = None
        live = base_hidden
        if live is None:
            raise RuntimeError("MEND PMRL needs the main-edit last-layer hidden state")
        if live.ndim == 3:
            mask = self._prepare_hf_visual_mask(batch)
            live = live[mask]
        return self._hidden_space_visual_pmrl(live)

    def edit(self, batch, condition=None, detach_history=False, return_factors=False,
             training=True, **kwargs):
        # Re-enable factor capture for the inner edit forward.  The trainer
        # disables it around checkpointed post-edit forwards so recomputation
        # cannot leave stale activation stacks behind.
        for module in self.model.modules():
            if hasattr(module, "_mend_capture"):
                module._mend_capture = True
        # Remove stale hook state from trainer pre-edit locality forwards.
        clear_mend_state(self.model, self.config.inner_params)
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower():
            outputs = self.model(batch)        
            if not isinstance(outputs, torch.Tensor):
                batch_labels = outputs.labels
                outputs = outputs.logits
            else:
                batch_labels = batch['labels']
            loss = self.edit_loss_fn(self.config, outputs, batch_labels, multimodal=True)["nll"]          
        elif "llava-onevision" in self.config.model_name.lower() or "qwen2-vl" in self.config.model_name.lower():
            extra_enabled = (
                getattr(self.config, "using_extra", False)
                and getattr(self.config, "using_lap", False)
                and getattr(self.config, "using_pmrl", False)
            )
            apply_extra = extra_enabled and (
                training or getattr(self.config, "mend_extra_at_eval", False)
            )
            model_inputs = dict(batch)
            model_inputs.pop("labels", None)
            captured_hidden = []
            capture_handle = None
            perturb_mask = None
            if apply_extra:
                perturb_mask = self._prepare_hf_visual_mask(batch)
                last_layer = self._hf_last_decoder_layer()
                capture_handle = last_layer.register_forward_hook(
                    lambda _m, _i, out, bucket=captured_hidden, mask=perturb_mask: bucket.append(
                        (out[0] if isinstance(out, tuple) else out)[mask].clone()
                    )
                )
            try:
                with profiling.section("edit.fwd_edit"):
                    outputs = self.model(**model_inputs)
            finally:
                if capture_handle is not None:
                    capture_handle.remove()
            with profiling.section("edit.loss"):
                loss = self._hf_target_loss(
                    outputs.logits, batch["labels"],
                    attention_mask=batch.get("attention_mask"),
                )
                if apply_extra:
                    if not captured_hidden:
                        raise RuntimeError("MEND extra hidden capture failed")
                    extra = self._hidden_space_visual_pmrl(captured_hidden.pop())
                    loss = loss + extra
                    del extra, captured_hidden, perturb_mask
            outputs = outputs.logits
        elif 'gpt' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            if not kwargs:
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
            else:
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"], **kwargs)["nll"]
        elif 'llama' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            if not kwargs:
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
            else:
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"], **kwargs)["nll"]
        elif 'baichuan' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"] 
        elif 'chatglm2' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]            
        elif 'internlm' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]  
        elif 'qwen' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]         
        elif 'mistral' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]  
        else:
            outputs = _logits(self.model(**batch))
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]

        names = set([n for n, p in self.model.named_parameters()])
        pset = set(self.config.inner_params)
        for p in pset:
            assert p in names, f"inner param {p} not in model"

        try:
            with profiling.section("edit.backward"):
                loss.backward()
            del loss, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            assert_mend_state_consumed(self.model, self.config.inner_params)
        except Exception:
            clear_mend_state(self.model, self.config.inner_params)
            raise

        def captured_factors(p):
            pairs = getattr(p, "__mend_pairs__", [])
            if pairs:
                return (
                    torch.cat([x.reshape(-1, x.shape[-1]) for x, _ in pairs], dim=0),
                    torch.cat([d.reshape(-1, d.shape[-1]) for _, d in pairs], dim=0),
                )
            return p.__x__, p.__delta__

        if self.config.shared:
            param_idx = (
                lambda n, p: self.shape_dict[self.get_shape(p)].index(n)
                if self.config.shared
                else None
            )  # noqa: E731
            transformed_factors = {
                n: self.mend[str(tuple(self.get_shape(p)))](
                    *captured_factors(p), param_idx(n, p)
                )
                for n, p in _inner_params(
                    self.model.named_parameters(), self.config.inner_params
                )
            }
        else:
            transformed_factors = {
                n: self.mend[n.replace(".", "#")](
                    *captured_factors(p)
                )
                for n, p in _inner_params(
                    self.model.named_parameters(), self.config.inner_params
                )
            }

        # Should be bi,bj->ji for nn.Linear, but GPT2 uses Conv1d instead...
        if isinstance(self.model, transformers.GPT2LMHeadModel):
            targ = "ij"
        else:
            targ = "ji"
        with profiling.section("edit.einsum"):
            mean_grads = {
                n: torch.einsum(f"bi,bj->{targ}", x, delta)
                for n, (x, delta) in transformed_factors.items()
            }
        del transformed_factors
        clear_mend_state(self.model, self.config.inner_params)

        info_dict = {}
        if getattr(self.config, "mend_log_grad_diagnostics", False):
            # Cheap connectivity diagnostics; full weight diagnostics can OOM.
            info_dict["diag/update_requires_grad"] = float(
                all(g.requires_grad for g in mean_grads.values())
            )
            info_dict["diag/update_grad_fn"] = float(
                all(g.grad_fn is not None for g in mean_grads.values())
            )
            info_dict["diag/pseudo_norm"] = float(
                sum(g.float().norm().item() for g in mean_grads.values())
            )
        if return_factors:
            info_dict["factors"] = locals().get("transformed_factors")
        # Full weight-sized (2x68M) magnitude/std/cosine diagnostics. They are
        # not part of the optimisation, allocate several weight-sized temporaries
        # and 12 blocking .item() syncs per step, and the trainer's averager
        # drops every "grad/" key (see RunningStatAverager exclude list) so the
        # values were never logged.  Keep them available behind their own flag
        # for one-off debugging, off by default so the step stays cheap.
        if getattr(self.config, "mend_log_weight_diagnostics", False):
            idx = 0
            for n, p in _inner_params(
                self.model.named_parameters(), self.config.inner_params
            ):
                info_dict[f"grad/true_mag{idx}"] = p.grad.norm(2).item()
                info_dict[f"grad/pseudo_mag{idx}"] = mean_grads[n].norm(2).item()
                info_dict[f"grad/true_std{idx}"] = p.grad.std().item()
                info_dict[f"grad/pseudo_std{idx}"] = mean_grads[n].std().item()
                info_dict[f"grad/diff{idx}"] = (p.grad - mean_grads[n]).norm(2).item()
                info_dict[f"grad/cos{idx}"] = F.cosine_similarity(
                    p.grad.reshape(-1), mean_grads[n].reshape(-1), dim=0
                ).item()
                idx += 1

        self.model.zero_grad()

        assert len(self.edit_lrs) == len(list(mean_grads.items()))
        updates = {n: lr * g for lr, (n, g) in zip(self.edit_lrs, mean_grads.items())}

        # HF multimodal models must keep the update tensors themselves in the
        # post-edit graph. Registering them on a monkey-patched nn.Module turns
        # the non-leaf tensors into detached Parameters and silently yields no
        # MEND outer gradients.
        if "llava-onevision" in self.config.model_name.lower() or "qwen2-vl" in self.config.model_name.lower():
            fast_params = {
                n: p + updates[n].to(p.dtype)
                for n, p in _inner_params(
                    self.model.named_parameters(), self.config.inner_params
                )
            }
            base_model = _FunctionalModel(self.model, fast_params)
            diagnostic_params = list(fast_params.values())
            if getattr(self.config, "mend_log_grad_diagnostics", False):
                info_dict["diag/fast_keys_applied"] = float(len(fast_params))
        else:
            base_model = self.model
            if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower():
                base_model = _make_functional(base_model, in_place=True)
            else:
                base_model = monkeypatch(base_model, in_place=True)
            new_params = []
            for n, p in base_model.named_parameters():
                mend_name = n if n in pset else f"model.{n}"
                new_params.append(p + updates[mend_name].to(p.dtype) if mend_name in pset else p)
            base_model.update_params(new_params)
            diagnostic_params = new_params

        if getattr(self.config, "mend_log_grad_diagnostics", False):
            info_dict["diag/fast_param_grad_fn"] = float(
                any(p.grad_fn is not None for p in diagnostic_params if torch.is_tensor(p))
            )
        # Keep a dict-aware MEND wrapper around the functional HF model. Build
        # it through EditableModel only (no hooks/parameter reinitialization),
        # then attach the shared transform and learning-rate parameters.
        edited_model = object.__new__(MEND)
        nn.Module.__init__(edited_model)
        EditableModel.__init__(
            edited_model, base_model, self.config, self.model_constructor
        )
        # Keep all original MEND hooks on the functional model. They must
        # capture the post-edit factors only for a subsequent edit; removing
        # them here also removes the module identity used by the meta-graph.
        edited_model.model.handles = getattr(self.model, "handles", [])
        edited_model.mend = self.mend
        edited_model.edit_lrs = self.edit_lrs
        edited_model.shape_dict = self.shape_dict
        if getattr(self.config, "mend_log_grad_diagnostics", False):
            info_dict["diag/shared_mend"] = 1.0
            info_dict["diag/mend_requires_grad"] = float(
                next(self.mend.parameters()).requires_grad
            )
        return edited_model, info_dict


if __name__ == "__main__":
    import types

    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

    config = types.SimpleNamespace()
    config.inner_params = [
        "transformer.h.9.mlp.c_fc.weight",
        "transformer.h.9.mlp.c_proj.weight",
        "transformer.h.10.mlp.c_fc.weight",
        "transformer.h.10.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
    config.edit_lr = 0.0001

    # config.mend = types.SimpleNamespace()
    config.n_hidden = 1
    config = config.__dict__

    mend = MEND(model, config, lambda: copy.deepcopy(model)).cuda()
    import pdb

    pdb.set_trace()
    mend.load_state_dict(torch.load("test_state.pt"))
    x = torch.arange(20).view(1, 20).cuda() + 1000
    orig_logits = mend(x)
    edited = mend.edit(x, masks=torch.ones_like(x), labels=x)
    post_logits = mend(x)

    assert torch.allclose(orig_logits, post_logits)

    orig_param = [
        p
        for (n, p) in mend.model.named_parameters()
        if n == config.inner_params[-1]
    ][0]
    edited_param = [
        p
        for (n, p) in edited.model.named_parameters()
        if n == config.inner_params[-1]
    ][0]

    LOG.info((orig_param - edited_param).abs().max())
    edited.eval()
    LOG.info(
        mend(x, labels=x).loss,
        edited(x, labels=x).loss,
        edited.edit_loss_fn(edited(x).logits, x)["nll"],
    )
    edited2 = edited.edit(x, masks=torch.ones_like(x), labels=x)
    LOG.info(
        mend(x, labels=x).loss, edited(x, labels=x).loss, edited2(x, labels=x).loss
    )


def monkeypatch(
    module: _torch.nn.Module,
    device: _typing.Optional[_torch.device] = None,
    copy_initial_weights: bool = True,
    track_higher_grads: bool = True,
    in_place: bool = False,
) -> _MonkeyPatchBase:
    r"""Create a monkey-patched stateless version of a module.
    This function produces a monkey-patched version of a module, and returns a
    copy of its parameters for use as fast weights. Where the original module
    or any of its submodules have state (e.g. batch norm), this will be copied
    too, but further updates (e.g. during inner loop training) will cause these
    to diverge without changing the state of the original module.
    Args:
        module: a ``torch.nn.Module`` subclass instance.
        device (optional): a device to cast the fast weights and state to.
        copy_initial_weights: if True, the weights of the patched module are
            copied to form the initial weights of the patched module, and thus
            are not part of the gradient tape when unrolling the patched module.
            If this is set to False, the actual module weights will be the
            initial weights of the patched module. This is useful when doing
            MAML, for example.
        track_higher_grads: if True, during unrolled optimization the graph be
            retained, and the fast weights will bear grad funcs, so as to permit
            backpropagation through the optimization process. Setting this to
            False allows ``monkeypatch`` to be used in "test mode", without
            potentially tracking higher order gradients. This can be useful when
            running the training loop at test time, e.g. in k-shot learning
            experiments, without incurring a significant memory overhead.
    Returns:
        ``fmodule``: a "stateless" version of the original module, for which calls
        to forward take the additional kwarg-only parameter ``params``, which
        should be a list of torch tensors requiring gradients, ideally
        provided by this function (see below) or by an update step from one
        of the optimizers in ``higher.optim``.
    """

    def encapsulator(fmodule: _MonkeyPatchBase, module: _torch.nn.Module) -> None:
        if copy_initial_weights and not in_place:
            params = _utils.get_func_params(module, device=device)
        elif in_place:
            params = [
                p if device is None else p.to(device) for p in module.parameters()
            ]
        else:  # Standard behavior
            params = [
                p.clone() if device is None else p.clone().to(device)
                for p in module.parameters()
            ]
        buffer_sync(module, fmodule, device)
        fmodule.update_params(params)

    fmodule = make_functional(module, encapsulator=encapsulator)
    fmodule.track_higher_grads = track_higher_grads

    return fmodule
