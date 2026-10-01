import glob
import json
import logging
import os
import shutil
import tempfile
import time

import torch
import copy
from .losses import kl_loc_loss
from .utils import *
from omegaconf import OmegaConf
from .models import *
from torch.utils.data import Dataset, DataLoader
from ..util.alg_train_dict import ALG_TRAIN_DICT
import importlib
from .overfit_stop import cap_max_iters, overfit_stopper_from_config
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    atomic_torch_save,
    capture_rng_state,
    formatted_timestamp,
    initial_global_iter,
    loc_floored_score,
    restore_rng_state,
    safe_backward,
    time_delta_seconds,
)

LOG = logging.getLogger(__name__)


def _main_optimizer_parameters(model):
    """Exclude edit_lrs when they are managed by MultimodalTrainer.lr_opt."""
    edit_lrs = getattr(model, "edit_lrs", None)
    for parameter in model.outer_parameters():
        if parameter is not edit_lrs:
            yield parameter


class BaseTrainer:
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        LOG.info(f'Config: {config}')
        configured_device = getattr(config, "device", None)
        if isinstance(configured_device, int) or str(configured_device).isdigit():
            index = int(configured_device)
            config.device = torch.device(
                f"cuda:{index}" if torch.cuda.is_available() else "cpu"
            )
        elif configured_device is not None:
            config.device = torch.device(str(configured_device))
        model_ = get_model(config)
        if 'qwen2' in config.model_name.lower() or 'llava-onevision' in config.model_name.lower():
            model_.bfloat16()
        self.alg_module = ALG_TRAIN_DICT[config.alg.upper()]
        LOG.info(f"Loading class {config.alg.upper()} from module {self.alg_module}")
        self.model = self.alg_module(model_, config, lambda: copy.deepcopy(model_))

        self.config = config

        if config.train_base:
            self.original_model = self.model.model_constructor()
            self.original_model.load_state_dict(self.model.model.state_dict())
            self.original_model.to(self.config.device)
        else:
            self.original_model = self.model.model

        if self.config.model_parallel:
            self.config.device = self.model.model.device
        if not self.config.model_parallel and hasattr(self.config, 'device'):
            self.model.to(self.config.device)

        self.train_set = train_set
        self.val_set = val_set

        # SERAC_MULTI decodes the cached edit text with the same processor the
        # dataset used to build the batch. The editor sets this at eval time;
        # training has to do it here.
        processor = getattr(train_set, "tok", None)
        if processor is not None and not hasattr(self.model, "processor"):
            self.model.processor = processor

        if 'minigpt4' in self.config.model_name.lower() or 'blip2' in self.config.model_name.lower():
            collate_fn = train_set.collate_fn
        elif "llava-onevision" in self.config.model_name.lower() or "qwen2-vl" in self.config.model_name.lower():
            collate_fn = train_set.collate_fn
        elif 't5' in self.config.model_class.lower():
            collate_fn = train_set.collate_fn
        elif 'gpt' in self.config.model_class.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'llama' in self.config.model_class.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'automodel' in self.config.model_class.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'qwen' in self.config.model_name.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'mistral' in self.config.model_name.lower():
            collate_fn = train_set.collate_gpt_fn
        else:
            raise NotImplementedError(f'Model {self.config.model_class} not supported yet.')

        # DataLoader workers overlap the (CPU-only) image processing and
        # tokenisation of the next batches with GPU compute; with the default
        # of 0 workers the collate runs inline in the training process, i.e.
        # the GPU idles for ~0.3s per step on the IC pipeline.  Workers get
        # ``device=None`` from the collate (see CaptionDataset.collate_fn) and
        # the trainer moves the batch afterwards.
        num_workers = int(getattr(self.config, "dataloader_num_workers", 0) or 0)
        loader_kwargs = {}
        if num_workers > 0:
            loader_kwargs = dict(
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
            LOG.info(f"DataLoader workers: {num_workers} (prefetch_factor=2, pin_memory=True)")

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size,
                                       shuffle=True, collate_fn=collate_fn, **loader_kwargs)
        self.val_loader = DataLoader(val_set, batch_size=self.config.val_batch_size,
                                       shuffle=False, collate_fn=collate_fn, **loader_kwargs)

        if self.config.eval_only:
            # Eval once and quit
            self.config.max_iters = 0

        if not self.config.eval_only and self.config.alg!='MALMEN':
            self.OptimizerClass = getattr(torch.optim, config.opt)
            LOG.info(f"Building optimizer {self.OptimizerClass} with lr {config.lr}")
            self.opt = self.OptimizerClass(_main_optimizer_parameters(self.model), lr=config.lr)

        if config.archive is not None:
            archive, config.archive = load_archive(str(config.archive))
            self.model.load_state_dict(archive["model"])
            del archive["model"]
            if not self.config.eval_only:
                if self.config.alg=='MALMEN':
                    self.model.opt.load_state_dict(archive["opt"])
                else:
                    self.opt.load_state_dict(archive["opt"])
            del archive["opt"]

            self.archive = (
                archive  # Save for later to load e.g. lr_opt params if they exist
            )
        else:
            self.archive = None

        self.lr_opt = None
        self.global_iter = initial_global_iter(self.archive)

        # # outfiles
        # with open(os.getcwd() + "/config.json", "w") as f:
        #     json.dump(OmegaConf.to_container(config), f)

        model_dir = os.path.join(config.results_dir, "models", config.alg)
        if not (self.config.debug and not self.config.save) and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        safe_model_name = self.config.model_name.split("/")[-1]  # Make sure no slashes
        self.save_path = f"{model_dir}/{safe_model_name}"

        if self.archive is not None and self.archive.get("start_time"):
            self.start_time = self.archive["start_time"]
        else:
            self.start_time = formatted_timestamp()
        if self.archive is not None:
            restore_rng_state(self.archive.get("rng_state"))

    def _checkpoint_payload(self, stats=None):
        stopper = getattr(self, "stopper", None)
        return {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict() if self.config.alg != "MALMEN" else self.model.opt.state_dict(),
            "lr_opt": self.lr_opt.state_dict() if self.lr_opt is not None else None,
            "val_stats": stats,
            "start_time": self.start_time,
            "elapsed_time": time_delta_seconds(self.start_time),
            "step": self.global_iter,
            "stopper": stopper.state_dict() if stopper is not None else None,
            "overfit_stopper": (
                self.overfit_stopper.state_dict()
                if getattr(self, "overfit_stopper", None) is not None
                else None
            ),
            "rng_state": capture_rng_state(),
        }

    def save_state(self, stats):
        if (self.config.debug and not self.config.save) or self.config.eval_only:
            return

        obj = self._checkpoint_payload(stats)
        LOG.info(f"Saving model to {self.save_path}")

        if os.path.exists(self.save_path):
            bk_path = f"{self.save_path}.bk"
            LOG.info(f"Moving old archive to {bk_path}")
            os.replace(self.save_path, bk_path)

        atomic_torch_save(obj, self.save_path)
        LOG.info("Write complete.")

    def save_prevalidation_state(self):
        """Persist a recoverable MEND state before memory-heavy validation."""
        if (self.config.debug and not self.config.save) or self.config.eval_only:
            return
        path = f"{self.save_path}.prevalidation"
        LOG.info(f"Saving pre-validation model to {path}")
        atomic_torch_save(self._checkpoint_payload(None), path)
        LOG.info("Pre-validation write complete.")

    def save_periodic(self):
        """Write ``.last`` and a numbered step snapshot; keep the newest 3 steps."""
        if (self.config.debug and not self.config.save) or self.config.eval_only:
            return
        payload = self._checkpoint_payload(None)
        last_path = f"{self.save_path}.last"
        step_path = f"{self.save_path}.step{self.global_iter}"
        LOG.info(f"Saving periodic checkpoint to {last_path} and {step_path}")
        atomic_torch_save(payload, last_path)
        atomic_torch_save(payload, step_path)
        pattern = f"{self.save_path}.step"
        numbered = []
        for path in glob.glob(pattern + "*"):
            suffix = path[len(pattern):]
            if suffix.isdigit():
                numbered.append((int(suffix), path))
        numbered.sort()
        for _, path in numbered[:-3]:
            try:
                os.unlink(path)
            except OSError:
                LOG.info(f"Could not prune old checkpoint {path}")
        LOG.info("Periodic write complete.")

    def _annotate_val_score(self, val_info):
        needed = (
            "edit/acc_val",
            "image_rephrase/acc_val",
            "loc/acc_val",
            "image_loc/acc_val",
        )
        if all(key in val_info for key in needed):
            t_floor = getattr(self.config, "baseline_t_loc", None)
            m_floor = getattr(self.config, "baseline_m_loc", None)
            val_info["acc/loc_floored_val"] = loc_floored_score(
                val_info, t_floor=t_floor, m_floor=m_floor
            )
        return val_info

    def _load_eval_archive(self):
        path = self.save_path
        if not os.path.exists(path):
            fallback = f"{self.save_path}.last"
            if os.path.exists(fallback):
                LOG.info(f"Best checkpoint missing; loading {fallback}")
                path = fallback
            else:
                LOG.info("No eval archive on disk; using in-memory weights")
                return
        if self.config.model_parallel:
            archive = torch.load(path)
        else:
            archive = torch.load(path, map_location="cpu", weights_only=False)
        LOG.info(
            f"Loading best model from step {archive['step']}, elapsed time {archive['elapsed_time']}"
        )
        if self.config.model_parallel:
            self.model.load_state_dict(archive["model"])
        else:
            self.model.to("cpu")
            self.model.load_state_dict(archive["model"])
            self.model.to(self.config.device)

    def echo(self, train_step, info_dict, pretty=False):
        if not self.config.silent:
            sep = "\n" if pretty else "; "

            def key_format(k):
                return k.ljust(20) if pretty else k

            LOG.info(f"Step {train_step}:")
            LOG.info(
                sep.join([f"{key_format(k)}: {v: 0.5f}" for k, v in info_dict.items()])
            )

    def run(self):
        averager = RunningStatAverager("train")
        stopper = EarlyStopper(
            self.config.early_stop_patience, self.config.early_stop_key
        )
        if self.archive is not None and self.archive.get("stopper"):
            try:
                stopper.load_state_dict(self.archive["stopper"])
            except Exception:
                LOG.info("Could not restore early-stopper state; starting a fresh stopper")
        self.stopper = stopper
        self.overfit_stopper = overfit_stopper_from_config(self.config)
        if self.overfit_stopper is not None:
            if self.archive is not None and self.archive.get("overfit_stopper"):
                try:
                    self.overfit_stopper.load_state_dict(self.archive["overfit_stopper"])
                except Exception:
                    LOG.info("Could not restore overfit-stopper state; starting fresh")
            hard_max = self.overfit_stopper.hard_max_step
            before = self.config.max_iters
            self.config.max_iters = cap_max_iters(self.config.max_iters, hard_max)
            LOG.info(
                f"OverfitStopper on key={self.overfit_stopper.key} "
                f"min_step={self.overfit_stopper.min_step} "
                f"hard_max={hard_max} (max_iters {before} -> {self.config.max_iters})"
            )
        if self.global_iter:
            LOG.info(f"Resuming from step {self.global_iter}")

        assert self.config.max_epochs is not None or self.config.max_iters is not None
        if self.config.max_epochs is not None:
            if self.config.max_iters is not None:
                self.config.max_iters = min(self.config.max_iters, self.config.max_epochs * len(self.train_set))
            else:
                self.config.max_iters = self.config.max_epochs * len(self.train_set)
            if self.config.alg == 'MALMEN':
                self.config.max_iters = math.ceil(self.config.max_iters / self.config.batch_size)
            LOG.info(f'MAX EPOCH: {self.config.max_epochs}, set max iters to {self.config.max_iters}')
        if self.config.alg == 'MALMEN':
            n_edits_step = math.ceil(self.config.n_edits / self.config.batch_size)
            if self.config.log_interval % n_edits_step:
                self.config.log_interval = (self.config.log_interval // n_edits_step) * n_edits_step if self.config.log_interval >= n_edits_step else n_edits_step
            if self.config.val_interval % n_edits_step:
                self.config.val_interval = (self.config.val_interval // n_edits_step) * n_edits_step if self.config.val_interval >= n_edits_step else n_edits_step
        self.epoches = round(float(self.config.max_iters) / (len(self.train_set) / self.config.batch_size))
        if self.epoches < 1:
            self.epoches = 1
        should_stop = False
        n_edits_batch = []
        for epoch in range(self.epoches):
            if should_stop:
                break
            for i, batch in enumerate(self.train_loader):
                if self.global_iter >= self.config.max_iters:
                    should_stop = True
                    break
                self.global_iter += 1
                if not self.config.eval_only:
                    if self.config.alg == 'MALMEN':  
                        n_edits_batch.append(batch)
                        if len(n_edits_batch) == math.ceil(self.config.n_edits / self.config.batch_size):
                            train_info = self.model.train(n_edits_batch)
                            averager.add(train_info)
                            n_edits_batch = []
                    else:
                        train_info = self.train_step(batch)
                        averager.add(train_info)

                    if self.global_iter % self.config.log_interval == 0:
                        avg_info = averager.average()
                        averager.reset()
                        self.echo(self.global_iter, avg_info)
                        if self.overfit_stopper is not None:
                            decision = self.overfit_stopper.update(
                                self.global_iter, avg_info
                            )
                            if decision.is_best and self.global_iter >= self.overfit_stopper.min_step:
                                LOG.info(
                                    f"OverfitStopper new best {decision.best_value:.5f} "
                                    f"at step {decision.best_step}"
                                )
                                self.save_state(avg_info)
                            if decision.stop:
                                LOG.info(f"OverfitStopper: {decision.reason}")
                                self.save_periodic()
                                should_stop = True
                    save_every = int(getattr(self.config, "model_save_pt", 0) or 0)
                    if save_every > 0 and self.global_iter % save_every == 0:
                        self.save_periodic()
                    if should_stop:
                        break
                if self.global_iter % self.config.val_interval == 0:
                    if getattr(self.config, "checkpoint_before_validation", False):
                        self.save_prevalidation_state()
                    if self.config.alg == 'MALMEN':
                        val_info = self.model.valid(config=self.config, loader=self.val_loader, val_set=self.val_set, steps=self.config.val_steps)
                    else:
                        val_info = self.validate(steps=self.config.val_steps)
                    val_info = self._annotate_val_score(val_info)
                    self.echo(self.global_iter, val_info)
                    is_best = stopper.update(self.global_iter, val_info)
                    if is_best:
                        self.save_state(val_info)
                    if stopper.should_stop():
                        LOG.info(
                            f"No decrease in {self.config.early_stop_key} for {self.config.early_stop_patience} steps"
                        )
                        should_stop = True
                        break

        if not self.config.eval_only:
            LOG.info(f"Training complete after {self.global_iter} steps.")

        if not self.config.final_eval:
            return

        if not self.config.eval_only:
            if (not self.config.debug) or self.config.save:
                self._load_eval_archive()

        val_steps = self.config.val_steps if self.config.debug else None
        if self.config.alg == 'MALMEN':
            val_info = self.model.valid(log=True, steps=val_steps, config=self.config, loader=self.val_loader, val_set=self.val_set)
        else:
            val_info = self.validate(log=True, steps=val_steps)
        self.echo(self.global_iter, val_info, pretty=True)

        if self.config.results_dir is not None:
            results_path = f"{self.config.results_dir}/results.json"
        else:
            results_path = f"{os.getcwd()}/results.json"

        with open(results_path, "w") as f:
            json.dump(
                {"results": val_info}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)
