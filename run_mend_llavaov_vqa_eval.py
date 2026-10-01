"""Evaluate LLaVA-OV MEND on MMEdit VQA."""
import os, json
from pathlib import Path
import torch
from easyeditor import CaptionDataset, MultimodalTrainer
from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
    MENDMultimodalTrainingHparams,
)

def run_eval(config_path, archive_path, results_dir, eval_size=100):
    hparams = MENDMultimodalTrainingHparams.from_hparams(config_path)
    hparams.eval_only = True
    hparams.archive = archive_path
    hparams.results_dir = results_dir
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
        "/root/MMEdit/editing-data/vqa/vqa_eval.json",
        config=hparams, size=eval_size,
    )
    train = CaptionDataset(
        "/root/MMEdit/editing-data/vqa/vqa_train.json",
        config=hparams, size=1,
    )
    print(f"MEND_VQA_EVAL_START eval_size={len(val)} archive={archive_path}", flush=True)
    trainer = MultimodalTrainer(hparams, train, val)
    val_info = trainer.validate(log=True, steps=eval_size)
    print(f"\nMEND_VQA_EVAL_DONE", flush=True)
    for k, v in val_info.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
    out_dir = Path(hparams.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump({"results": {k: float(v) if isinstance(v, (torch.Tensor, float)) else v
                                for k, v in val_info.items()}}, f, indent=2)
    print(f"Results saved to {out_dir / 'result.json'}", flush=True)
    return val_info

def main():
    eval_size = int(os.environ.get("PMRL_MEND_EVAL_SIZE", "100"))
    print("=" * 60, flush=True)
    print("BASELINE VQA", flush=True)
    print("=" * 60, flush=True)
    run_eval(
        config_path="hparams/TRAINING/MEND/llavaov-7b-vqa.yaml",
        archive_path="results/MEND_LLAVAOV_BASELINE_VQA_META_TRAIN/models/MEND/llava-onevision",
        results_dir="results/MEND_LLAVAOV_VQA_BASELINE_N100",
        eval_size=eval_size,
    )
    print("\n" + "=" * 60, flush=True)
    print("ASAM VQA", flush=True)
    print("=" * 60, flush=True)
    run_eval(
        config_path="hparams/TRAINING/MEND/llavaov-7b-vqa-lap-pmrl.yaml",
        archive_path="results/MEND_LLAVAOV_LAP_PMRL_VQA_META_TRAIN/models/MEND/llava-onevision",
        results_dir="results/MEND_LLAVAOV_VQA_LAP_PMRL_N100",
        eval_size=eval_size,
    )

if __name__ == "__main__":
    main()