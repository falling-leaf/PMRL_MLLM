"""Evaluate Qwen2-VL MEND ASAM IC using the same validation method as training."""
import os
import json
from pathlib import Path

import torch

from easyeditor import CaptionDataset, MultimodalTrainer
from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
    MENDMultimodalTrainingHparams,
)


def main():
    config_path = "hparams/TRAINING/MEND/qwen2vl-7b-lap-pmrl.yaml"
    eval_size = int(os.environ.get("PMRL_MEND_EVAL_SIZE", "100"))

    hparams = MENDMultimodalTrainingHparams.from_hparams(config_path)
    hparams.eval_only = True
    hparams.mend_extra_at_eval = True
    hparams.qwen_max_pixels = 1280 * 28 * 28
    hparams.archive = os.environ.get(
        "PMRL_MEND_ARCHIVE",
        "results/MEND_QWEN2VL_LAP_PMRL_META_TRAIN/models/MEND/qwen2-vl",
    )
    hparams.results_dir = os.environ.get(
        "PMRL_MEND_RESULTS_DIR",
        "results/MEND_QWEN2VL_IC_ASAM_VISUAL_N100",
    )
    hparams.max_iters = 0
    hparams.val_steps = eval_size
    hparams.final_eval = True
    hparams.save = False
    hparams.silent = False
    hparams.verbose = True
    hparams.debug = False

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)

    val = CaptionDataset(
        "/root/MMEdit/editing-data/caption/caption_eval_edit.json",
        config=hparams,
        size=eval_size,
    )
    train = CaptionDataset(
        "/root/MMEdit/editing-data/caption/caption_train_edit.json",
        config=hparams,
        size=1,
    )

    print(f"MEND_ASAM_EVAL_START eval_size={len(val)} archive={hparams.archive}", flush=True)
    trainer = MultimodalTrainer(hparams, train, val)
    val_info = trainer.validate(log=True, steps=eval_size)

    print(f"\nMEND_ASAM_EVAL_DONE", flush=True)
    for k, v in val_info.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)

    out_dir = Path(hparams.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump({"results": {k: float(v) if isinstance(v, (torch.Tensor, float)) else v
                                for k, v in val_info.items()}}, f, indent=2)
    print(f"Results saved to {out_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()