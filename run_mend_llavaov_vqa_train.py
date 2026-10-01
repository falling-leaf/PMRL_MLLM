"""Meta-train LLaVA-OV MEND on MMEdit VQA."""
import os
from pathlib import Path
import torch
from easyeditor import CaptionDataset, MultimodalTrainer
from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
    MENDMultimodalTrainingHparams,
)

def main():
    config_path = os.environ.get(
        "PMRL_MEND_HPARAMS", "hparams/TRAINING/MEND/llavaov-7b-vqa.yaml"
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
    if bool(getattr(hparams, "overfit_stop", False)):
        from easyeditor.trainer.overfit_stop import cap_max_iters
        hard_max = int(getattr(hparams, "overfit_stop_hard_max", 10000))
        hparams.max_iters = cap_max_iters(hparams.max_iters, hard_max)
    resume = os.environ.get("PMRL_MEND_RESUME")
    if resume:
        hparams.archive = resume
    elif not hparams.archive:
        last = (
            Path(hparams.results_dir) / "models" / "MEND" / "llava-onevision.last"
        )
        if last.exists():
            hparams.archive = str(last)
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
        f"MEND_META_TRAIN_START config={config_path} train={len(train)} "
        f"val={len(val)} extra={hparams.using_extra} asam={getattr(hparams, 'using_asam', False)} "
        f"archive={hparams.archive} views={hparams.num_rephrase}",
        flush=True,
    )
    MultimodalTrainer(hparams, train, val).run()
    print("MEND_META_TRAIN_DONE", flush=True)

if __name__ == "__main__":
    main()