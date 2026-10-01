from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .algs import profiling
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .algs.outer_asam import apply_outer_asam_grads
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    dict_to,
    formatted_timestamp,
    move_to_device,
    safe_backward,
    time_delta_seconds,
)

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        extra_on = (
            bool(getattr(self.config, "using_extra", False))
            and bool(getattr(self.config, "using_lap", False))
            and bool(getattr(self.config, "using_pmrl", False))
        )
        if bool(getattr(self.config, "using_asam", False)) and extra_on:
            raise ValueError(
                "Outer ASAM and inner LAP+PMRL cannot be enabled together"
            )
        self._asam_window = []

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None and self.archive.get("lr_opt") is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def _batch_to_device(self, batch):
        """Place a batch produced by DataLoader workers onto the training device.

        With ``dataloader_num_workers > 0`` the collate function runs inside a
        forked worker, which must not touch CUDA, so the batch arrives on the
        CPU.  With workers disabled the tensors are already on the device and
        this is a cheap device check.
        """
        head = batch["edit_inner"]
        if isinstance(head, torch.Tensor):
            need_move = head.device.type == "cpu"
        elif torch.is_tensor(head.get("input_ids")):
            need_move = head["input_ids"].device.type == "cpu"
        else:
            need_move = False
        return move_to_device(batch, self.config.device) if need_move else batch

    def _maybe_flush_gpu_cache(self):
        """Return cached blocks to the driver only when the device is tight.

        ``torch.cuda.empty_cache()`` after every step hands ~60GB of activation
        blocks back to the driver and forces the next step to re-``cudaMalloc``
        them; the caching allocator re-uses them far faster.  The run peaks far
        below the device capacity, so keep the flush as a safety valve for the
        case where free memory actually drops below the configured threshold.
        """
        if not torch.cuda.is_available():
            return
        threshold_gb = float(
            getattr(self.config, "gpu_cache_free_threshold_gb", 8.0) or 0.0
        )
        if threshold_gb <= 0:
            return
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < threshold_gb * (1 << 30):
            torch.cuda.empty_cache()

    def edit_step(self, batch, training: bool):
        batch = self._batch_to_device(batch)
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            capture_modules = [m for m in self.model.modules() if hasattr(m, "_mend_capture")]
            for module in capture_modules:
                module._mend_capture = False
            try:
                with profiling.section("step.fwd_base_loc"):
                    base_outputs = self.model(batch["loc"])
                if not isinstance(base_outputs, torch.Tensor):
                    base_logits = base_outputs.logits
                else:
                    base_logits = base_outputs
                with profiling.section("step.fwd_base_loc_image"):
                    base_image_outputs = self.model(batch["loc_image"])
                if not isinstance(base_image_outputs, torch.Tensor):
                    base_image_logits = base_image_outputs.logits
                else:
                    base_image_logits = base_image_outputs
            finally:
                for module in capture_modules:
                    module._mend_capture = True

        start = time.time()
        edited_model, model_info = self.model.edit(
            batch["edit_inner"], batch["cond"], training=training
        )
        edit_time = time.time() - start

        def post_forward(payload):
            """Run the edited model.

            SERAC's counterfactual mixture is already a full sequence, and the
            chunked KL backward replays a checkpointed forward once per chunk,
            which turned one step into a few hundred forwards. SERAC does not
            need that recomputation; MEND still does.
            """
            if not training or self.config.alg.upper().startswith("SERAC"):
                return edited_model(payload)
            capture_modules = list(self.model.modules())
            for module in capture_modules:
                if hasattr(module, "_mend_capture"):
                    module._mend_capture = False
            dummy = torch.ones((), device=self.config.device, requires_grad=True)
            try:
                return torch.utils.checkpoint.checkpoint(
                    lambda _dummy: edited_model(payload).logits,
                    dummy, use_reentrant=False,
                )
            finally:
                for module in capture_modules:
                    if hasattr(module, "_mend_capture"):
                        module._mend_capture = True

        with torch.set_grad_enabled(training):
            # Editing loss
            with profiling.section("step.fwd_post_edit_outer"):
                post_edit_outputs = post_forward(batch["edit_outer"])
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
                # 检查输出对象是否有 labels 属性，如果没有则从 batch 中获取
                post_batch_labels = getattr(post_edit_outputs, 'labels', None)
                if post_batch_labels is None:
                    post_batch_labels = batch["edit_outer"]["labels"]
            else:
                post_edit_logits = post_edit_outputs
                post_batch_labels = batch["edit_outer"]["labels"]

            # The inner-example forward below only feeds metrics: edit_loss_fn
            # is called on its logits under no_grad and no l_* term references
            # them, so keeping it out of the autograd graph cannot change a
            # single number (the tensor values are identical), it only stops
            # the step from building and holding a full-length graph for a
            # forward that is never differentiated.  ComprehendEditDataset
            # reuses this tensor as the image-rephrase logits, so that dataset
            # keeps the grad-enabled path.
            inner_feeds_loss = (
                self.train_set.__class__.__name__ == "ComprehendEditDataset"
            )
            if inner_feeds_loss:
                with profiling.section("step.fwd_post_edit_inner"):
                    inner_edit_outputs = edited_model(batch["edit_inner"])
            else:
                with profiling.section("step.fwd_post_edit_inner"):
                    with torch.no_grad():
                        inner_edit_outputs = edited_model(batch["edit_inner"])
            
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
                # 检查输出对象是否有 labels 属性，如果没有则从 batch 中获取
                inner_batch_labels = getattr(inner_edit_outputs, 'labels', None)
                if inner_batch_labels is None:
                    inner_batch_labels = batch["edit_inner"]["labels"]
            else:
                inner_edit_logits = inner_edit_outputs
                inner_batch_labels = batch["edit_inner"]["labels"]

            # rephrase image
            if self.train_set.__class__.__name__ == "ComprehendEditDataset":
                post_image_edit_logits = inner_edit_logits
                post_image_batch_labels = inner_batch_labels
            else:
                with profiling.section("step.fwd_post_edit_outer_image"):
                    post_image_edit_outputs = post_forward(batch["edit_outer_image"])
                if not isinstance(post_image_edit_outputs, torch.Tensor):
                    post_image_edit_logits = post_image_edit_outputs.logits
                    # 检查输出对象是否有 labels 属性，如果没有则从 batch 中获取
                    post_image_batch_labels = getattr(post_image_edit_outputs, 'labels', None)
                    if post_image_batch_labels is None:
                        post_image_batch_labels = batch["edit_outer_image"]["labels"]
                else:
                    post_image_edit_logits = post_image_edit_outputs
                    post_image_batch_labels = batch["edit_outer_image"]["labels"]
            l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)["nll"]
            l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)["nll"]
            # Collect useful metrics before releasing the large logits.  The
            # metric pass is non-differentiable and does not affect the loss.
            with torch.no_grad():
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels, multimodal=True)
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)
            post_requires_grad = float(post_edit_logits.requires_grad)
            post_grad_fn = float(post_edit_logits.grad_fn is not None)
            with profiling.section("step.fwd_post_loc"):
                post_base_outputs = post_forward(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
                kl_mask = batch["loc"].get("attention_mask", None)
            else:
                post_base_logits = post_base_outputs
                kl_mask = batch["loc"].get("attention_mask", None)
            if kl_mask is None:
                if post_base_logits.shape[:2] != base_logits.shape[:2]:
                    raise ValueError("Text locality logits changed shape without an attention mask")
                kl_mask = torch.ones(
                    post_base_logits.shape[:2], device=post_base_logits.device, dtype=torch.bool
                )
            if kl_mask.shape != post_base_logits.shape[:2]:
                raise ValueError("Text locality attention mask does not match logits")
            if self.config.alg.upper().startswith("SERAC"):
                # The prompt is ~3000 tokens but the edit only supervises the
                # answer. A KL over the whole sequence is a softmax over
                # millions of logits per step and changes nothing the loss
                # cares about, so score locality on the answer rows only.
                ans = batch["loc"].get("labels")
                if ans is not None and ans.shape == kl_mask.shape:
                    kl_mask = kl_mask.bool() & ans.ne(-100)

            with profiling.section("step.fwd_post_loc_image"):
                post_image_base_outputs = post_forward(batch["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
                kl_image_mask = batch["loc_image"].get("attention_mask", None)
            else:
                post_image_base_logits = post_image_base_outputs
                kl_image_mask = batch["loc_image"].get("attention_mask", None)
            if kl_image_mask is None:
                if post_image_base_logits.shape[:2] != base_image_logits.shape[:2]:
                    raise ValueError("Image locality logits changed shape without an attention mask")
                kl_image_mask = torch.ones(
                    post_image_base_logits.shape[:2], device=post_image_base_logits.device, dtype=torch.bool
                )
            if kl_image_mask.shape != post_image_base_logits.shape[:2]:
                raise ValueError("Image locality attention mask does not match logits")
            if self.config.alg.upper().startswith("SERAC"):
                ans = batch["loc_image"].get("labels")
                if ans is not None and ans.shape == kl_image_mask.shape:
                    kl_image_mask = kl_image_mask.bool() & ans.ne(-100)

            with profiling.section("step.kl_loc"):
                kl_chunk_size = int(getattr(self.config, "kl_chunk_size", 32))
                l_loc = kl_loc_loss(
                    base_logits.detach(), post_base_logits,
                    mask=kl_mask, chunk_size=kl_chunk_size,
                )
                l_image_loc = kl_loc_loss(
                    base_image_logits.detach(), post_image_base_logits,
                    mask=kl_image_mask, chunk_size=kl_chunk_size,
                )

        # if l_edit.isnan():
        #     print("l_edit is nan")
        #     print("input: ", batch["edit_outer"]['text_input'])
        # elif l_image_edit.isnan():
        #     print("l_image_edit is nan")
        #     print("input: ", batch["edit_outer_image"]['text_input'])
        # elif l_loc.isnan():
        #     print("l_loc is nan")
        #     print("input: ", batch["loc"]['text_input'])
        # elif l_image_loc.isnan():
        #     print("l_image_loc is nan")
        #     print("input: ", batch["loc_image"]['text_input'])

        objective_requires_grad = (
            self.config.cedit * l_edit
            + self.config.cloc * (l_loc + l_image_loc)
            + self.config.iedit * l_image_edit
        ).requires_grad
        info_dict = {}
        if getattr(self.config, "mend_log_grad_diagnostics", False):
            info_dict["diag/post_requires_grad"] = post_requires_grad
            info_dict["diag/post_grad_fn"] = post_grad_fn
            info_dict["diag/l_total_requires_grad"] = float(objective_requires_grad)

        edit_loss_value = float(post_edit_dict["nll"].detach().cpu())
        image_edit_loss_value = float(image_rephrase_edit_dict["nll"].detach().cpu())
        edit_outer_parameters = None
        if training and self.config.alg != 'ft':
            # Keep edit gradients until locality has been differentiated; the
            # MEND fast-weight graph is shared by all objective terms.
            edit_outer_parameters = list(self.model.outer_parameters())

        # Locality agreement metrics.  post_base_logits carries autograd
        # history (it is part of l_loc's graph), so a softmax + top-k over a
        # [1, ~3k, 152k] tensor would keep the softmax output and a sorted copy
        # alive for backward -- several GB per step for pure monitoring
        # numbers.  The indices returned under no_grad are bit-identical.
        with torch.no_grad():
            if base_logits.shape != post_base_logits.shape:
                raise ValueError("Text locality base/post logits have different shapes")
            if base_image_logits.shape != post_image_base_logits.shape:
                raise ValueError("Image locality base/post logits have different shapes")
            # Softmax is monotonic, so top-k over logits gives the same indices
            # without allocating a full-vocabulary probability tensor.
            text_valid = kl_mask.bool().reshape(-1)
            image_valid = kl_image_mask.bool().reshape(-1)
            text_base = base_logits.reshape(-1, base_logits.shape[-1])[text_valid]
            text_post = post_base_logits.reshape(-1, post_base_logits.shape[-1])[text_valid]
            image_base = base_image_logits.reshape(-1, base_image_logits.shape[-1])[image_valid]
            image_post = post_image_base_logits.reshape(-1, post_image_base_logits.shape[-1])[image_valid]
            if text_base.numel() == 0 or image_base.numel() == 0:
                raise ValueError("Locality attention mask contains no valid positions")

            text_post_top_k = torch.topk(text_post, k=1, dim=-1).indices
            text_base_top_k = torch.topk(text_base, k=1, dim=-1).indices
            image_post_top_k = torch.topk(image_post, k=10, dim=-1).indices
            image_base_top_k = torch.topk(image_base, k=10, dim=-1).indices
            loc_acc = (text_post_top_k == text_base_top_k).float().mean()
            image_loc_acc = (image_post_top_k == image_base_top_k).float().mean()

        # Materialize metrics as host scalars before releasing the graph;
        # RunningStatAverager must not retain CUDA tensors between steps.
        loc_loss_value = float(l_loc.detach().cpu())
        image_loc_loss_value = float(l_image_loc.detach().cpu())
        info_dict['loss/edit'] = edit_loss_value
        info_dict['loss/image_edit'] = image_edit_loss_value
        info_dict['loss/loc'] = loc_loss_value
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["time/edit"] = edit_time
        info_dict["loc/acc"] = loc_acc
        info_dict["image_loc/acc"] = image_loc_acc
        l_base = torch.tensor(0.0)
        total_loss_tensor = self.config.cloc * (l_loc + l_image_loc) + self.config.cbase * l_base

        info_dict["loss/total"] = float(
            self.config.cedit * edit_loss_value
            + self.config.cloc * (loc_loss_value + image_loc_loss_value)
            + self.config.iedit * image_edit_loss_value
        )
        info_dict["loss/total_edit"] = info_dict["loss/total"]
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            info_dict["memory/free"] = free_bytes
        # Convert metric tensors to host scalars before releasing the graph;
        # RunningStatAverager must not retain CUDA tensors between steps.
        if training and self.config.alg != 'ft':
            with profiling.section("step.meta_backward_total"):
                safe_backward(
                    self.config.cedit * l_edit
                    + self.config.cloc * (l_loc + l_image_loc)
                    + self.config.iedit * l_image_edit,
                    edit_outer_parameters, self.config.accumulate_bs,
                    allow_unused=True,
                )
        info_dict = {
            k: (float(v.detach().cpu()) if torch.is_tensor(v) and v.ndim == 0 else v)
            for k, v in {**info_dict, **model_info}.items()
        }
        del model_info
        del inner_edit_outputs
        del post_base_outputs, post_image_base_outputs
        del inner_edit_logits
        del l_loc, l_image_loc
        del post_base_logits, post_image_base_logits
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del post_forward
        self._maybe_flush_gpu_cache()
        profiling.dump(LOG, steps=getattr(self, "global_iter", None))

        return total_loss_tensor, edit_loss_value, loc_loss_value, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )
        using_asam = bool(getattr(self.config, "using_asam", False))
        if using_asam:
            self._asam_window.append(batch)
            ready = len(self._asam_window) >= int(self.config.accumulate_bs)
        else:
            ready = self.global_iter % self.config.accumulate_bs == 0

        if ready:
            params = list(self.model.outer_parameters())
            allow_nonfinite = using_asam
            grad = torch.nn.utils.clip_grad_norm_(
                params,
                self.config.grad_clip,
                error_if_nonfinite=not allow_nonfinite,
            )
            info_dict["grad"] = float(grad) if torch.isfinite(grad) else float("nan")
            info_dict["diag/asam_skipped"] = 0.0

            if using_asam:
                first_finite = bool(
                    torch.isfinite(torch.as_tensor(info_dict["grad"])).item()
                )
                if first_finite:
                    first_grads = [
                        None if parameter.grad is None else parameter.grad.detach().clone()
                        for parameter in params
                    ]
                    window = list(self._asam_window)

                    def second_pass():
                        for replay in window:
                            self.edit_step(replay, training=True)

                    exclude = []
                    if bool(getattr(self.config, "asam_exclude_edit_lrs", True)):
                        edit_lrs = getattr(self.model, "edit_lrs", None)
                        if edit_lrs is not None:
                            exclude = [edit_lrs]
                    result = apply_outer_asam_grads(
                        params,
                        first_grads,
                        second_pass,
                        epsilon=float(getattr(self.config, "asam_epsilon", 0.002)),
                        rho=float(getattr(self.config, "asam_scale_rho", 0.1)),
                        replace=bool(getattr(self.config, "asam_replace", True)),
                        exclude=exclude,
                    )
                    info_dict["diag/asam_skipped"] = 1.0 if result["skipped"] else 0.0
                    grad = torch.nn.utils.clip_grad_norm_(
                        params,
                        self.config.grad_clip,
                        error_if_nonfinite=False,
                    )
                    info_dict["grad"] = float(grad) if torch.isfinite(grad) else float("nan")
                else:
                    info_dict["diag/asam_skipped"] = 1.0
            self._asam_window = []

            if torch.isfinite(torch.as_tensor(info_dict["grad"])):
                self.opt.step()
                if self.lr_opt is not None:
                    self.lr_opt.step()
            else:
                LOG.info("Skipping optimizer step because clipped grad is non-finite")
            self.opt.zero_grad(set_to_none=True)

            if self.lr_opt is not None:
                self.lr_opt.zero_grad(set_to_none=True)
                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f"lr/lr{lr_idx}"] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"

        LOG.info(
          f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}"
        )

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in enumerate(self.val_loader):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats