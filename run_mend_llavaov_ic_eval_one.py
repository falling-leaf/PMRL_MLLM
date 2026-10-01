"""N=100 independent evaluation of the paused LLaVA-OV MEND LAP+PMRL (ASAM) IC run.

Two checkpoints of the same run, same config, same 100-case eval set, same
seed - so the two rows are directly paired with each other and with the
baseline (`results/MEND_LLAVAOV_BASELINE_META_TRAIN/results.json`, which used
the identical protocol: archive-loaded validation over `caption_eval_edit`,
`val_steps=100`, shift-corrected edit accuracy).

    PMRL_MEND_EVAL_ARCHIVE=... PMRL_MEND_EVAL_OUT=... python run_mend_llavaov_ic_eval_one.py
"""

import json
import os
from pathlib import Path

import torch

from easyeditor import CaptionDataset, MultimodalTrainer
from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
    MENDMultimodalTrainingHparams,
)

CONFIG = os.environ.get(
    "PMRL_MEND_EVAL_CONFIG", "hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-graphfix-final2.yaml"
)
DEFAULT_ARCHIVE = ("results/MEND_LLAVAOV_LAP_PMRL_IC_GRAPHFIX_FINAL2/models/MEND/"
                   "llava-onevision.step15000")
DEFAULT_OUT = "results/MEND_LLAVAOV_IC_LAP_PMRL_GRAPHFIX_STEP15000_N100"


def main():
    config = CONFIG
    archive = os.environ.get("PMRL_MEND_EVAL_ARCHIVE", DEFAULT_ARCHIVE)
    out = os.environ.get("PMRL_MEND_EVAL_OUT", DEFAULT_OUT)
    step = int(os.environ.get("PMRL_MEND_EVAL_STEP", "15000"))
    eval_size = int(os.environ.get("PMRL_MEND_EVAL_SIZE", "100"))

    h = MENDMultimodalTrainingHparams.from_hparams(config)
    h.eval_only = True
    h.archive = archive
    h.results_dir = out
    h.max_iters = 0
    h.val_steps = eval_size
    h.final_eval = True
    h.save = False
    h.silent = False
    h.verbose = True
    h.debug = False
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)

    val = CaptionDataset(
        "/root/MMEdit/editing-data/caption/caption_eval_edit.json", config=h, size=eval_size
    )
    train = CaptionDataset(
        "/root/MMEdit/editing-data/caption/caption_train_edit.json", config=h, size=1
    )
    if len(val) != eval_size:
        raise RuntimeError(f"Expected {eval_size} eval records, got {len(val)}")
    print(
        f"MEND_IC_EVAL_START archive={archive} step={step} eval_size={len(val)} "
        f"config={config} shift=True",
        flush=True,
    )
    info = MultimodalTrainer(h, train, val).validate(log=True, steps=eval_size)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "archive": str(Path(archive).resolve()),
        "checkpoint_step": step,
        "config": config,
        "split": "caption_eval_edit",
        "sample_count": len(val),
        "validation_mode": "archive_loaded_validation_shift_corrected",
        "results": {
            k: (float(v) if isinstance(v, (float, torch.Tensor)) else v) for k, v in info.items()
        },
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2))
    print("MEND_IC_EVAL_DONE", flush=True)
    for k, v in info.items():
        print(f"  {k}: {v}", flush=True)
    print(f"RESULT={out_dir.resolve() / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
