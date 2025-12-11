from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        # 检查是否配置了sub_device用于双卡部署
        use_sub_device = (hasattr(self.config, 'sub_device') and 
                         self.config.sub_device is not None and 
                         self.config.sub_device != self.config.device)
        
        if use_sub_device:
            # 确保sub_device是字符串格式
            if isinstance(self.config.sub_device, int):
                sub_device_str = f'cuda:{self.config.sub_device}'
            elif isinstance(self.config.sub_device, str):
                if not self.config.sub_device.startswith('cuda'):
                    sub_device_str = f'cuda:{self.config.sub_device}'
                else:
                    sub_device_str = self.config.sub_device
            else:
                sub_device_str = None
                use_sub_device = False
        else:
            sub_device_str = None

        with torch.no_grad():
            base_outputs = self.model(batch["loc"])
            if not isinstance(base_outputs, torch.Tensor):
                base_logits = base_outputs.logits
            else:  
                base_logits = base_outputs
                
            base_image_outputs = self.model(batch["loc_image"])
            if not isinstance(base_image_outputs, torch.Tensor):
                base_image_logits = base_image_outputs.logits
            else:
                base_image_logits = base_image_outputs
        
        # Do the edit

        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start

        # 辅助函数：将batch移到指定设备
        def move_batch_to_device(batch_dict, device):
            """将batch字典中的所有tensor移到指定设备"""
            if isinstance(batch_dict, dict):
                return {k: move_batch_to_device(v, device) if isinstance(v, (dict, torch.Tensor)) else v 
                       for k, v in batch_dict.items()}
            elif isinstance(batch_dict, torch.Tensor):
                return batch_dict.to(device)
            else:
                return batch_dict

        with torch.set_grad_enabled(training):
            # 如果使用sub_device，将edited_model的前向传播移到sub_device
            # 注意：edited_model是functional模型，我们需要将输入移到sub_device
            if use_sub_device:
                # 将edited_model相关的batch移到sub_device
                edit_outer_batch = move_batch_to_device(batch["edit_outer"], sub_device_str)
                post_edit_outputs = edited_model(edit_outer_batch)
                # 提取logits并移回主设备
                if not isinstance(post_edit_outputs, torch.Tensor):
                    post_edit_logits = post_edit_outputs.logits.to(self.config.device)
                    post_batch_labels = getattr(post_edit_outputs, 'labels', None)
                    if post_batch_labels is None:
                        post_batch_labels = batch["edit_outer"]["labels"]
                    else:
                        post_batch_labels = post_batch_labels.to(self.config.device) if isinstance(post_batch_labels, torch.Tensor) else post_batch_labels
                else:
                    post_edit_logits = post_edit_outputs.to(self.config.device)
                    post_batch_labels = batch["edit_outer"]["labels"]
            else:
                post_edit_outputs = edited_model(batch["edit_outer"])
                if not isinstance(post_edit_outputs, torch.Tensor):
                    post_edit_logits = post_edit_outputs.logits
                    # 检查输出对象是否有 labels 属性，如果没有则从 batch 中获取
                    post_batch_labels = getattr(post_edit_outputs, 'labels', None)
                    if post_batch_labels is None:
                        post_batch_labels = batch["edit_outer"]["labels"]
                else:
                    post_edit_logits = post_edit_outputs
                    post_batch_labels = batch["edit_outer"]["labels"]

            if use_sub_device:
                edit_inner_batch = move_batch_to_device(batch["edit_inner"], sub_device_str)
                inner_edit_outputs = edited_model(edit_inner_batch)
                if not isinstance(inner_edit_outputs, torch.Tensor):
                    inner_edit_logits = inner_edit_outputs.logits.to(self.config.device)
                    inner_batch_labels = getattr(inner_edit_outputs, 'labels', None)
                    if inner_batch_labels is None:
                        inner_batch_labels = batch["edit_inner"]["labels"]
                    else:
                        inner_batch_labels = inner_batch_labels.to(self.config.device) if isinstance(inner_batch_labels, torch.Tensor) else inner_batch_labels
                else:
                    inner_edit_logits = inner_edit_outputs.to(self.config.device)
                    inner_batch_labels = batch["edit_inner"]["labels"]
            else:
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
                if use_sub_device:
                    edit_outer_image_batch = move_batch_to_device(batch["edit_outer_image"], sub_device_str)
                    post_image_edit_outputs = edited_model(edit_outer_image_batch)
                    if not isinstance(post_image_edit_outputs, torch.Tensor):
                        post_image_edit_logits = post_image_edit_outputs.logits.to(self.config.device)
                        post_image_batch_labels = getattr(post_image_edit_outputs, 'labels', None)
                        if post_image_batch_labels is None:
                            post_image_batch_labels = batch["edit_outer_image"]["labels"]
                        else:
                            post_image_batch_labels = post_image_batch_labels.to(self.config.device) if isinstance(post_image_batch_labels, torch.Tensor) else post_image_batch_labels
                    else:
                        post_image_edit_logits = post_image_edit_outputs.to(self.config.device)
                        post_image_batch_labels = batch["edit_outer_image"]["labels"]
                else:
                    post_image_edit_outputs = edited_model(batch["edit_outer_image"])
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
            
            # Collect some useful metrics
            with torch.no_grad():
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels, multimodal=True)
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)
            
            if use_sub_device:
                loc_batch = move_batch_to_device(batch["loc"], sub_device_str)
                post_base_outputs = edited_model(loc_batch)
                if not isinstance(post_base_outputs, torch.Tensor):
                    post_base_logits = post_base_outputs.logits.to(self.config.device)
                    kl_mask = getattr(post_base_outputs, 'attention_mask', None)
                    if kl_mask is None:
                        kl_mask = batch["loc"].get("attention_mask", None)
                        if kl_mask is None:
                            kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(self.config.device)
                        else:
                            kl_mask = kl_mask.to(self.config.device) if isinstance(kl_mask, torch.Tensor) else kl_mask
                    else:
                        kl_mask = kl_mask.to(self.config.device) if isinstance(kl_mask, torch.Tensor) else kl_mask
                else:
                    post_base_logits = post_base_outputs.to(self.config.device)
                    kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(self.config.device)
            else:
                post_base_outputs = edited_model(batch["loc"])
                if not isinstance(post_base_outputs, torch.Tensor):
                    post_base_logits = post_base_outputs.logits
                    kl_mask = getattr(post_base_outputs, 'attention_mask', None)
                    if kl_mask is None:
                        kl_mask = batch["loc"].get("attention_mask", None)
                        if kl_mask is None:
                            kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)
                else:
                    post_base_logits = post_base_outputs
                    kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

            if use_sub_device:
                loc_image_batch = move_batch_to_device(batch["loc_image"], sub_device_str)
                post_image_base_outputs = edited_model(loc_image_batch)
                if not isinstance(post_image_base_outputs, torch.Tensor):
                    post_image_base_logits = post_image_base_outputs.logits.to(self.config.device)
                    kl_image_mask = getattr(post_image_base_outputs, 'attention_mask', None)
                    if kl_image_mask is None:
                        kl_image_mask = batch["loc_image"].get("attention_mask", None)
                        if kl_image_mask is None:
                            kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(self.config.device)
                        else:
                            kl_image_mask = kl_image_mask.to(self.config.device) if isinstance(kl_image_mask, torch.Tensor) else kl_image_mask
                    else:
                        kl_image_mask = kl_image_mask.to(self.config.device) if isinstance(kl_image_mask, torch.Tensor) else kl_image_mask
                else:
                    post_image_base_logits = post_image_base_outputs.to(self.config.device)
                    kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(self.config.device)
            else:
                post_image_base_outputs = edited_model(batch["loc_image"])
                if not isinstance(post_image_base_outputs, torch.Tensor):
                    post_image_base_logits = post_image_base_outputs.logits
                    kl_image_mask = getattr(post_image_base_outputs, 'attention_mask', None)
                    if kl_image_mask is None:
                        kl_image_mask = batch["loc_image"].get("attention_mask", None)
                        if kl_image_mask is None:
                            kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(post_image_base_logits.device)
                else:
                    post_image_base_logits = post_image_base_outputs
                    kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(post_image_base_logits.device)
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
                kl_image_mask = getattr(post_image_base_outputs, 'attention_mask', None)
                if kl_image_mask is None:
                    kl_image_mask = batch["loc_image"].get("attention_mask", None)
                    if kl_image_mask is None:
                        kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(post_image_base_logits.device)
            else:
                post_image_base_logits = post_image_base_outputs
                kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(base_image_logits.device)

            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
            l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

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

        if self.config.alg == "SERAC_MULTI":
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + self.config.iedit * l_image_edit
        else:
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc + l_image_loc) + self.config.iedit * l_image_edit
        

        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        # Text locality
        post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
        base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

        # Image locality
        post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
        base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices

        info_dict = {}
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["time/edit"] = edit_time
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        
        # 如果使用sub_device，也记录sub_device的显存使用
        if use_sub_device:
            info_dict["memory/alloc_max_sub"] = torch.cuda.max_memory_allocated(device=sub_device_str)
            info_dict["memory/res_max_sub"] = torch.cuda.max_memory_reserved(device=sub_device_str)
        
        info_dict = {**info_dict, **model_info}
        
        # 清理显存：删除不需要的中间变量
        if use_sub_device:
            # 清理sub_device上的缓存
            torch.cuda.empty_cache()
            # 清理主设备上的缓存
            torch.cuda.empty_cache()

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

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