"""Pre-flight image integrity scan for the LLaVA-OV IC caption datasets.

Motivation: the first 30 000-step LLaVA-OV MEND LAP+PMRL (ASAM) IC run died at
step 1000 with ``OSError: image file is truncated`` raised inside a DataLoader
worker's collate, i.e. a corrupt image referenced by the training data.  This
scan walks exactly the images the run touches (first ``--size`` records of the
train json, all records of the eval json, the same three fields
``image`` / ``image_rephrase`` / ``m_loc`` that ``CaptionDataset.__init__``
feeds to the vision processor), fully decodes each one, and reports:

* files that fail a strict ``PIL.Image.load()`` (would abort a run),
* for those, whether the decode is salvageable with
  ``ImageFile.LOAD_TRUNCATED_IMAGES = True`` and how many pixels survive,
* byte sizes vs. the directory median (a size outlier usually means a
  truncated download/copy).

Usage:
    python tests/scan_caption_image_integrity.py                 # train 1000 + eval all
    python tests/scan_caption_image_integrity.py --size 1000 --limit-eval 1000
    python tests/scan_caption_image_integrity.py --json out.json # machine-readable

Exit code 0 = no corrupt file found, 1 = at least one unusable image.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile

TRAIN_JSON = "/root/MMEdit/editing-data/caption/caption_train_edit.json"
EVAL_JSON = "/root/MMEdit/editing-data/caption/caption_eval_edit.json"
VIS_ROOT = "/root/MMEdit/images"
REPHRASE_ROOT = "/root/MMEdit/images"
FIELDS = ("image", "image_rephrase", "m_loc")


def strict_decode(path):
    """Open+fully decode, returns (ok, error, size_px)."""
    try:
        with Image.open(path) as im:
            im.load()
            return True, None, im.size
    except Exception as exc:  # noqa: BLE001 - report anything PIL raises
        return False, f"{type(exc).__name__}: {exc}", None


def salvaged_decode(path):
    """Retry with LOAD_TRUNCATED_IMAGES, which is what a robustness patch does."""
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as im:
            im.load()
            return True, None, im.size
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1000, help="train records used by the run")
    ap.add_argument("--limit-eval", type=int, default=0, help="0 = all eval records")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    jobs = []
    train = json.load(open(TRAIN_JSON))
    eval_ = json.load(open(EVAL_JSON))
    for rec in train[: args.size]:
        for f in FIELDS:
            jobs.append((f"train:{f}", os.path.join(VIS_ROOT, rec[f])))
    for rec in (eval_[: args.limit_eval] if args.limit_eval else eval_):
        for f in FIELDS:
            jobs.append((f"eval:{f}", os.path.join(VIS_ROOT, rec[f])))
    print(f"scanning {len(jobs)} image references "
          f"(train[:{args.size}] + eval[:{args.limit_eval or 'all'}]) ...", flush=True)

    seen, bad, sizes = {}, [], {}
    for i, (tag, path) in enumerate(jobs):
        if path in seen:
            continue
        seen[path] = True
        if not os.path.exists(path):
            bad.append({"tag": tag, "path": path, "error": "missing file"})
            continue
        n = os.path.getsize(path)
        sizes.setdefault(os.path.dirname(path), []).append(n)
        ok, err, size = strict_decode(path)
        if not ok:
            sok, serr, ssize = salvaged_decode(path)
            bad.append({"tag": tag, "path": path, "bytes": n, "error": err,
                        "salvageable": sok, "salvage_error": serr,
                        "salvaged_size": ssize})
            print(f"  BAD  {tag:<20} {path}  {n} B  {err}", flush=True)
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(jobs)}", flush=True)

    med = {d: sorted(v)[len(v) // 2] for d, v in sizes.items() if v}
    for b in bad:
        d = os.path.dirname(b["path"])
        if b.get("bytes") and d in med:
            b["dir_median_bytes"] = med[d]
    print(f"\nchecked {len(seen)} unique files; {len(bad)} unusable")
    if bad:
        print("corrupt/missing:")
        for b in bad:
            print(f"  {b['path']}  {b.get('bytes')} B  {b['error']}"
                  f"  salvageable={b.get('salvageable')}")
    kinds = Counter(b["error"].split(":")[0] for b in bad)
    out = {"scan": {"train_size": args.size, "limit_eval": args.limit_eval,
                    "files_checked": len(seen), "unusable": len(bad),
                    "error_kinds": dict(kinds)},
           "bad": bad, "dir_median_bytes": med}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print("wrote", args.json)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
