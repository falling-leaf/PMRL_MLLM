"""Diagnose the LLaVA-OV IC caption DataLoader (multi-worker collate).

Background: the run `run_logs/mend_llavaov_ic_full_asam_20260912.log` died at
step ~1004 with ``OSError: image file is truncated`` raised in a DataLoader
worker's collate, while a full PIL decode of every referenced file
(`tests/scan_caption_image_integrity.py`) finds nothing corrupt.  That points at
the loader/collate path rather than at the data, so this script exercises it
without a model:

    python tests/diagnose_caption_dataloader.py --workers 4 --epochs 2
    python tests/diagnose_caption_dataloader.py --workers 0 --epochs 2

It builds the real ``CaptionDataset`` (size from --size) and a DataLoader with
the same kwargs the trainer uses, then iterates ``--epochs`` full epochs while
logging every PIL file the workers open (per-process log file, so the last line
before a crash names the image involved).  Exit code 1 if the loader raised.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageFile  # noqa: E402

OUT = Path("/tmp/diag_dataloader")
OUT.mkdir(exist_ok=True)
# recomputed in worker_init_fn so forked workers get their own file
LOG_PATH = OUT / f"pil_open_pid{os.getpid()}_{os.getpid()}.log"

_orig_open = Image.open
_orig_load = ImageFile.ImageFile.load


def _log(msg):
    with open(LOG_PATH, "a") as fh:
        fh.write(msg + "\n")


def open_hook(fp, *a, **kw):
    try:
        _log(f"OPEN {fp}")
    except Exception:
        pass
    return _orig_open(fp, *a, **kw)


def load_hook(self, *a, **kw):
    try:
        _log(f"LOAD {getattr(self, 'filename', '?')}")
    except Exception:
        pass
    return _orig_load(self, *a, **kw)


Image.open = open_hook
ImageFile.ImageFile.load = load_hook


def worker_init(_):
    global LOG_PATH
    wid = __import__("torch").utils.data.get_worker_info().id
    LOG_PATH = OUT / f"pil_open_pid{os.getpid()}_w{wid}.log"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--config", default="hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-graphfix-final2.yaml")
    ap.add_argument("--json", default=str(OUT / "result.json"))
    args = ap.parse_args()

    import torch  # noqa: F401
    from easyeditor import CaptionDataset
    from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
        MENDMultimodalTrainingHparams,
    )

    hp = MENDMultimodalTrainingHparams.from_hparams(args.config)
    t0 = time.time()
    ds = CaptionDataset(
        "/root/MMEdit/editing-data/caption/caption_train_edit.json", config=hp, size=args.size
    )
    init_s = time.time() - t0
    n = args.size * args.epochs
    print(f"dataset ready in {init_s:.1f}s ({len(ds)} items); workers={args.workers}, "
          f"target batches={n:.0f}", flush=True)

    from torch.utils.data import DataLoader

    kw = {}
    if args.workers:
        kw = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=ds.collate_fn,
                        worker_init_fn=worker_init if args.workers else None, **kw)

    done, failed, err = 0, False, None
    t0 = time.time()
    try:
        # one pass over the loader == one epoch; iterate explicitly so the
        # epoch boundary (where the real run died) is actually crossed
        for epoch in range(int(args.epochs)):
            for _ in loader:
                done += 1
                if done % 200 == 0:
                    print(f"  batch {done}/{n:.0f} ({time.time()-t0:.0f}s)", flush=True)
            print(f"  epoch {epoch+1} done, {done} batches", flush=True)
    except Exception:
        failed, err = True, traceback.format_exc()
        print("!!! loader raised:\n" + err[-1500:], flush=True)

    # per-worker open history tail, to name the image involved in a failure
    tails = {}
    for f in sorted(OUT.glob("pil_open_*.log")):
        lines = f.read_text().splitlines()
        tails[f.name] = lines[-3:]
    res = {"workers": args.workers, "epochs": args.epochs, "size": args.size,
           "items": len(ds), "init_s": round(init_s, 1), "batches_done": done,
           "elapsed_s": round(time.time() - t0, 1), "failed": failed,
           "error": (err or "")[-800:], "open_log_tails": tails}
    Path(args.json).write_text(json.dumps(res, indent=2))
    print(f"\nbatches_done={done} elapsed={res['elapsed_s']}s failed={failed}")
    print("last PIL ops per process:", json.dumps(tails, indent=1)[:600])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
