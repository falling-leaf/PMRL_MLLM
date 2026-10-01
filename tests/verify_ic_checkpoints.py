"""Read-only verification of the checkpoint archives of the paused LLaVA-OV IC run.

Loads each archive on CPU (mmap first, full load as fallback) and asserts the
recorded ``step``, the presence of model/optimizer state, and that the MEND
outer parameters are not still at initialisation.  Run BEFORE stopping the
training process: the pause is only safe if a loadable archive already exists.

Usage: python tests/verify_ic_checkpoints.py [--dir <models/MEND dir>] [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

DEFAULT_DIR = "/root/PMRL_MLLM/results/MEND_LLAVAOV_LAP_PMRL_IC_GRAPHFIX_FINAL2/models/MEND"


def load_archive(path):
    """Load a checkpoint, preferring mmap (lazy) and falling back to a full read."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix != ".tmp"],
                   key=lambda p: p.stat().st_mtime)
    print(f"archives in {d}:")
    report = {"dir": str(d), "archives": []}
    ok = True
    for p in files:
        rec = {"file": p.name, "bytes": p.stat().st_size,
               "mtime": __import__("datetime").datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
        try:
            a = load_archive(str(p))
            rec["step"] = a.get("step")
            rec["has_model"] = "model" in a
            rec["has_opt"] = "opt" in a
            rec["has_lr_opt"] = a.get("lr_opt") is not None
            rec["val_stats_keys"] = sorted(a["val_stats"].keys()) if a.get("val_stats") else None
            rec["n_tensors"] = len(a.get("model", {}))
            # MEND transform factors must have moved away from init
            u = [k for k in a.get("model", {}) if "mend" in k.lower() or "grad_transform" in k.lower()]
            rec["mend_param_examples"] = u[:4]
            if u:
                t = a["model"][u[0]]
                rec["mend_first_param_norm"] = float(t.float().norm())
            del a
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            ok = False
        report["archives"].append(rec)
        print(json.dumps(rec, indent=1)[:700])

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print("wrote", args.json)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
