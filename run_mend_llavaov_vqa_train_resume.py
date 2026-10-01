"""Resume LLaVA-OV MEND meta-training on MMEdit VQA from step 15000 checkpoint.

Optimizations (all safe, no training impact):
- torch.backends.cudnn.benchmark = True  (auto-tune cuDNN kernels)
- torch.set_float32_matmul_precision('high')  (Tensor Cores for float32, slight numerical diff)
"""
import os
from pathlib import Path
import torch
from easyeditor import CaptionDataset, MultimodalTrainer
from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
    MENDMultimodalTrainingHparams,
)

def main():
    # --- Performance optimizations (safe, no training impact) ---
    torch.backends.cudnn.benchmark = True
    # 'high' uses Tensor Cores for float32 matmuls with full float32 accumulation.
    # On Blackwell (compute 12.0) this is significantly faster with negligible
    # numerical difference vs pure float32.
    torch.set_float32_matmul_precision('high')

    config_path = os.environ.get(
        "PMRL_MEND_HPARAMS", "hparams/TRAINING/MEND/llavaov-7b-vqa-resume.yaml"
    )
    train_size = int(os.environ.get("PMRL_MEND_TRAIN_SIZE", "1000"))
    val_size = int(os.environ.get("PMRL_MEND_VAL_SIZE", "100"))
    max_iters_override = os.environ.get("PMRL_MEND_MAX_ITERS")
    val_interval_override = os.environ.get("PMRL_MEND_VAL_INTERVAL")

    hparams = MENDMultimodalTrainingHparams.from_hparams(config_path)
    if max_iters_override is not None:
        hparams.max_iters = int(max_iters_override)
    if val_interval_override is not None:
        hparams.val_interval = int(val_interval_override)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)

    train = CaptionDataset(
        "/root/MMEdit/editing-data/vqa/vqa_train.json",
        config=hparams, size=train_size,
    )
    val = CaptionDataset(
        "/root/MMEdit/editing-data/vqa/vqa_eval.json",
        config=hparams, size=val_size,
    )
    if len(train) != train_size or len(val) != val_size:
        raise RuntimeError(
            f"Expected train/val {train_size}/{val_size}, got {len(train)}/{len(val)}"
        )
    Path(hparams.results_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"MEND_META_TRAIN_RESUME config={config_path} train={len(train)} "
        f"val={len(val)} archive={hparams.archive} "
        f"max_iters={hparams.max_iters} val_interval={hparams.val_interval}",
        flush=True,
    )
    trainer = MultimodalTrainer(hparams, train, val)
    trainer.run()
    print("MEND_META_TRAIN_RESUME_DONE", flush=True)

if __name__ == "__main__":
    main()