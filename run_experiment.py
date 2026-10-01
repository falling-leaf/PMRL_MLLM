#!/usr/bin/env python3
"""Unified launcher for PMRL_MLLM multimodal editing experiments.

The launcher intentionally keeps MEND's existing entrypoints untouched.  WISE,
Transformer-Patcher, and UniKE share the evaluation/editing path here; MEND is
forwarded to the repository's existing train/eval scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PMRL_DATA_DIR", ROOT.parent / "MMEdit"))
DATASETS = {
    "IC": DATA_ROOT / "editing-data" / "caption" / "caption_eval_edit.json",
    "VQA": DATA_ROOT / "editing-data" / "vqa" / "vqa_eval.json",
}
TRAIN_DATASETS = {
    "IC": DATA_ROOT / "editing-data" / "caption" / "caption_train_edit.json",
    "VQA": DATA_ROOT / "editing-data" / "vqa" / "vqa_train.json",
}
MODELS = {"blip2", "minigpt4", "qwen2vl", "llavaov"}
METHODS = {"wise", "tpatch", "unike", "mend", "preedit", "serac", "ike", "ft"}
TASKS = {"IC", "VQA"}


def _first_existing(candidates) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of these configs exist:\n  " + "\n  ".join(str(c) for c in candidates))


def default_config(method: str, model: str, task: str, mode: str) -> Path:
    task_l = task.lower()
    if method == "wise":
        suffix = "baseline" if mode == "baseline" else "lap_pmrl"
        return _first_existing([
            ROOT / "hparams" / "WISE" / f"{model}_{task_l}_{suffix}.yaml",
            ROOT / "hparams" / "WISE" / f"{model}.yaml",
        ])
    if method == "tpatch":
        folder = ROOT / "hparams" / "Transformer-Patcher"
        if mode == "baseline":
            return _first_existing([
                folder / f"{model}_{task_l}_tpatch.yaml",
                folder / f"{model}_{task_l}_tpatch_baseline.yaml",
                folder / f"{model}_tpatch_baseline.yaml",
            ])
        return _first_existing([
            folder / f"{model}_{task_l}_tpatch_asam_w025.yaml",
            folder / f"{model}_{task_l}_tpatch_asam_initial_w025.yaml",
            folder / f"{model}_tpatch_asam_initial_w025.yaml",
            folder / f"{model}_{task_l}_tpatch_pmrl.yaml",
        ])
    if method == "unike":
        suffix = "baseline" if mode == "baseline" else "asam"
        folder = ROOT / "hparams" / "UniKE"
        # BLIP-2 uses the dedicated editor; the other three use UniKE-Simplified.
        return _first_existing([
            folder / f"{model}_{task_l}_unike_{suffix}.yaml",
            folder / f"{model}_{task_l}_unike_simplified_{suffix}.yaml",
            folder / "blip2_ic_unike_online_simplified.yaml" if model == "blip2" and mode == "baseline" else folder / f"{model}_{task_l}_unike_{suffix}.yaml",
        ])
    if method == "mend":
        return mend_config(model, task, mode)
    if method in {"preedit", "serac", "ike", "ft"}:
        # These four are plain baselines. `mode` is ignored on purpose so an
        # ASAM flag cannot silently select a different config.
        folder = {"preedit": "preedit", "serac": "SERAC", "ike": "IKE", "ft": "FT"}[method]
        if method == "serac":
            name = model
        elif model in {"llavaov", "qwen2vl"}:
            name = "llavaov_7b" if model == "llavaov" else f"{model}_7b"
        else:
            # BLIP2 / MiniGPT4 configs are not 7B-stem files.
            name = model
        return ROOT / "hparams" / folder / f"{name}.yaml"
    raise ValueError(f"Unsupported method: {method}")


def scalar(value):
    try:
        import torch
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
    except ImportError:
        pass
    return float(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--task", choices=sorted(TASKS), type=str.upper, required=True)
    parser.add_argument("--mode", choices=["baseline", "asam"], default="baseline")
    parser.add_argument("--config", type=Path, help="Hyperparameter YAML; overrides the method default")
    parser.add_argument("--size", type=int, default=100, help="Number of effective evaluation records")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase", choices=["edit", "train", "eval"], default="edit")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config/command without loading models")
    return parser


def mend_config(model: str, task: str, mode: str) -> Path:
    """Training YAML. BLIP-2 and MiniGPT-4 share one file across IC and VQA."""
    folder = ROOT / "hparams" / "TRAINING" / "MEND"
    if model in {"blip2", "minigpt4"}:
        return folder / f"{model}.yaml"
    if mode == "asam":
        return _first_existing([
            folder / f"{model}-7b-lap-pmrl.yaml",
            folder / f"{model}-7b-{task.lower()}-lap-pmrl.yaml",
            folder / f"{model}-7b.yaml",
        ])
    return _first_existing([
        folder / f"{model}-7b.yaml",
        folder / f"{model}-7b-{task.lower()}.yaml",
    ])


def mend_script(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    if args.model in {"blip2", "minigpt4"}:
        raise FileNotFoundError(
            "BLIP-2 and MiniGPT-4 MEND have a training config "
            f"({mend_config(args.model, args.task, args.mode)}) but no dedicated "
            "run_mend_* script. Train them with MultimodalTrainer the same way "
            "as run_mend_qwen2vl_ic_train.py, passing PMRL_MEND_HPARAMS."
        )
    model_name = {"qwen2vl": "qwen2vl", "llavaov": "llavaov"}[args.model]
    task = args.task.lower()
    if args.phase == "train":
        script = ROOT / f"run_mend_{model_name}_{task}_train.py"
    else:
        # Existing evaluation scripts are deliberately reused instead of edited.
        if model_name == "qwen2vl" and task == "ic":
            script = ROOT / ("run_mend_qwen2vl_ic_asam_eval.py" if args.mode == "asam" else "run_mend_qwen2vl_ic_eval.py")
        elif model_name == "qwen2vl" and task == "vqa":
            script = ROOT / "run_mend_qwen2vl_vqa_eval.py"
        else:
            script = ROOT / f"run_mend_{model_name}_{task}_eval.py"
    if not script.exists():
        raise FileNotFoundError(f"Existing MEND entrypoint not found: {script}")
    env = os.environ.copy()
    env["PMRL_MEND_HPARAMS"] = str(args.config or mend_config(args.model, args.task, args.mode))
    env["PMRL_DATA_DIR"] = str(DATA_ROOT)
    env["PMRL_MEND_EVAL_SIZE"] = str(args.size)
    if args.output_dir:
        env["PMRL_OUTPUT_DIR"] = str(args.output_dir)
    return [sys.executable, str(script)], env


def run_edit(args: argparse.Namespace, config: Path) -> dict:
    import torch
    from easyeditor import CaptionDataset, MultimodalEditor, VQADataset
    if args.method == "wise":
        from easyeditor import WISEMultimodalHyperParams as HyperParams
    elif args.method == "tpatch":
        from easyeditor.models.transformer_patcher import TransformerPatcherMultimodalHyperParams as HyperParams
    elif args.method == "preedit":
        from easyeditor.models.ike import PreEditHyperParams as HyperParams
    elif args.method == "ike":
        from easyeditor.models.ike import IKEHyperParams as HyperParams
    elif args.method == "ft":
        from easyeditor.models.ft import FTHyperParams as HyperParams
    elif args.method == "serac":
        if args.phase == "train":
            from easyeditor import SERACMultimodalTrainingHparams as HyperParams
        else:
            from easyeditor.models.serac import SERACMultimodalHparams as HyperParams
    else:
        from easyeditor.models.unike_blip2 import UniKEBLIP2HyperParams as BlipHyperParams
        from easyeditor.models.unike_simplified import UniKESimplifiedHyperParams as SimpleHyperParams
        HyperParams = BlipHyperParams if args.model == "blip2" else SimpleHyperParams

    hp = HyperParams.from_hparams(str(config))
    dataset_cls = CaptionDataset if args.task == "IC" else VQADataset
    dataset_path = DATASETS[args.task]
    dataset = dataset_cls(str(dataset_path), config=hp, size=args.size)
    if len(dataset) != args.size:
        raise RuntimeError(f"Expected {args.size} effective records, got {len(dataset)}")
    output = args.output_dir or ROOT / "results" / f"{args.method}_{args.model}_{args.task}_{args.mode}"
    output.mkdir(parents=True, exist_ok=True)
    editor = MultimodalEditor.from_hparams(hp)
    started = time.time()
    metrics, _, _ = editor.edit_dataset(ds=dataset, keep_original_weight=True, verbose=True)
    if len(metrics) != args.size:
        raise RuntimeError(f"Expected {args.size} metric records, got {len(metrics)}")
    source = {"rewrite_acc": "rewrite_acc", "rephrase_acc": "rephrase_acc", "rephrase_image_acc": ("image_rephrase_acc", "rephrase_image_acc"), "locality_acc": "locality_acc", "multimodal_locality_acc": "multimodal_locality_acc"}
    def _pick(row, names):
        post = row["post"]
        for name in (names if isinstance(names, tuple) else (names,)):
            if name in post:
                return scalar(post[name])
        raise KeyError(f"{names} not in {list(post)}")
    aggregate = {key: mean(_pick(row, value) for row in metrics) for key, value in source.items()}
    aggregate["gen_avg"] = mean([aggregate["rephrase_acc"], aggregate["rephrase_image_acc"]])
    aggregate["loc_avg"] = mean([aggregate["locality_acc"], aggregate["multimodal_locality_acc"]])
    result = {"repository": "PMRL_MLLM", "method": args.method, "model": args.model, "task": args.task, "mode": args.mode, "config": str(config), "dataset": str(dataset_path), "sample_count": args.size, "seed": args.seed, "wall_seconds": time.time() - started, "metrics": aggregate}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT=" + json.dumps(result))
    return result


def run_serac_train(args: argparse.Namespace, config: Path) -> int:
    """Train SERAC with the existing multimodal trainer. No ASAM hooks."""
    from easyeditor import (
        CaptionDataset,
        MultimodalTrainer,
        SERACMultimodalTrainingHparams,
        VQADataset,
    )

    hp = SERACMultimodalTrainingHparams.from_hparams(str(config))
    dataset_cls = CaptionDataset if args.task == "IC" else VQADataset
    train_path = TRAIN_DATASETS[args.task]
    eval_path = DATASETS[args.task]
    train_ds = dataset_cls(str(train_path), config=hp, size=args.size)
    val_ds = dataset_cls(str(eval_path), config=hp, size=min(args.size, 20))
    trainer = MultimodalTrainer(config=hp, train_set=train_ds, val_set=val_ds)
    trainer.run()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.method in {"preedit", "serac", "ike", "ft"} and args.mode == "asam":
        raise SystemExit(
            f"{args.method} is a reference baseline and has no ASAM config. "
            "Re-run with --mode baseline."
        )
    config = args.config or default_config(args.method, args.model, args.task, args.mode)
    if not config.exists():
        raise FileNotFoundError(f"Config does not exist: {config}")
    if args.method == "mend":
        if args.model in {"blip2", "minigpt4"}:
            print(json.dumps({
                "method": "mend",
                "model": args.model,
                "task": args.task,
                "mode": args.mode,
                "config": str(config),
                "note": "No run_mend_* script for this backbone. Use MultimodalTrainer with this YAML.",
            }, indent=2))
            return 0
        command, env = mend_script(args)
        print(json.dumps({"command": command, "config": env.get("PMRL_MEND_HPARAMS")}, indent=2))
        if args.dry_run:
            return 0
        return subprocess.call(command, cwd=ROOT, env=env)
    if args.method == "serac" and args.phase == "train" and args.config is None:
        config = ROOT / "hparams" / "TRAINING" / "SERAC" / f"{args.model}.yaml"
    print(json.dumps({"method": args.method, "model": args.model, "task": args.task, "mode": args.mode, "phase": args.phase, "config": str(config)}, indent=2))
    if args.dry_run:
        return 0
    if args.method == "serac" and args.phase == "train":
        seed_everything(args.seed)
        return run_serac_train(args, config)
    seed_everything(args.seed)
    run_edit(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
