"""Diagnose why the baseline archive's recorded val_stats (edit/acc 0.712) does not
reproduce under current code (0.0205), while the ASAM archive reproduces bitwise.

Hypothesis: the edit magnitude applied at validation time differs because
``edit_lrs`` restored from the archive (1.43e-5 / 2.41e-5) are ~7x smaller than
the config's ``edit_lr`` (1e-4).  This script reports, for a given archive:
  * the edit_lrs actually used by the editor after the archive is loaded,
  * the edit_lrs stored inside the archive,
  * validation edit/acc under (a) restored edit_lrs, (b) config edit_lr.

Usage: python tests/diagnose_eval_edit_lrs.py <archive> <config> [--force-lr 1e-4]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from easyeditor import CaptionDataset, MultimodalTrainer  # noqa: E402
from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (  # noqa: E402
    MENDMultimodalTrainingHparams,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("config")
    ap.add_argument("--force-lr", type=float, default=None)
    ap.add_argument("--size", type=int, default=100)
    args = ap.parse_args()

    h = MENDMultimodalTrainingHparams.from_hparams(args.config)
    h.eval_only = True
    h.archive = args.archive
    h.results_dir = "/tmp/diag_eval_lrs"
    h.max_iters = 0
    h.val_steps = args.size
    h.final_eval = True
    h.save = False
    h.silent = True
    h.debug = False
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)

    val = CaptionDataset("/root/MMEdit/editing-data/caption/caption_eval_edit.json",
                         config=h, size=args.size)
    train = CaptionDataset("/root/MMEdit/editing-data/caption/caption_train_edit.json",
                           config=h, size=1)
    tr = MultimodalTrainer(h, train, val)
    used = tr.model.edit_lrs.detach().float().cpu().tolist()
    stored = torch.load(args.archive, map_location="cpu", weights_only=False,
                        mmap=True)["model"]["edit_lrs"].tolist()
    print(f"config edit_lr      : {h.edit_lr}")
    print(f"archive edit_lrs    : {stored}")
    print(f"editor edit_lrs used: {used}")
    if args.force_lr is not None:
        with torch.no_grad():
            tr.model.edit_lrs.fill_(args.force_lr)
        print(f"FORCED edit_lrs -> {tr.model.edit_lrs.detach().cpu().tolist()}")
    info = tr.validate(log=False, steps=args.size)
    out = {k: float(v) for k, v in info.items()
           if k in ("edit/acc_val", "image_rephrase/acc_val", "loc/acc_val",
                    "image_loc/acc_val", "loss/edit_val", "loss/loc_val")}
    print("validation:", json.dumps(out, indent=1))
    Path("/tmp/diag_eval_lrs").mkdir(exist_ok=True)
    tag = f"forced{args.force_lr}" if args.force_lr is not None else "restored"
    Path(f"/tmp/diag_eval_lrs/{Path(args.archive).name}_{tag}.json").write_text(
        json.dumps({"archive": args.archive, "config": args.config,
                    "stored_edit_lrs": stored, "used_edit_lrs": used,
                    "forced_edit_lr": args.force_lr, "results": out}, indent=2))


if __name__ == "__main__":
    main()
