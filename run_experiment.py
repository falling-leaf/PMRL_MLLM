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
DATASETS = {
    "IC": ROOT.parent / "MMEdit" / "editing-data" / "caption" / "caption_eval_edit.json",
    "VQA": ROOT.parent / "MMEdit" / "editing-data" / "vqa" / "vqa_eval.json",
}
MODELS = {"blip2", "minigpt4", "qwen2vl", "llavaov"}
METHODS = {"wise", "tpatch", "unike", "mend"}
TASKS = {"IC", "VQA"}


def default_config(method: str, model: str, task: str, mode: str) -> Path:
    if method == "wise":
        suffix = "baseline" if mode == "baseline" else "lap_pmrl"
        return ROOT / "hparams" / "WISE" / f"{model}_{task.lower()}_{suffix}.yaml"
    if method == "tpatch":
        if mode == "baseline":
            candidates = [
                ROOT / "hparams" / "Transformer-Patcher" / f"{model}_{task.lower()}_tpatch.yaml",
                ROOT / "hparams" / "Transformer-Patcher" / f"{model}_{task.lower()}_tpatch_baseline.yaml",
            ]
            if model == "qwen2vl":
                candidates.insert(0, ROOT / "hparams/Transformer-Patcher" / f"qwen2vl_{task.lower()}_tpatch_baseline.yaml")
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"No default T-Patcher baseline config for {model}/{task}")
        candidates = [
            ROOT / "hparams" / "Transformer-Patcher" / f"{model}_{task.lower()}_tpatch_asam_initial_w025.yaml",
            ROOT / "hparams" / "Transformer-Patcher" / f"{model}_{task.lower()}_tpatch_asam_w025.yaml",
        ]
        if model == "qwen2vl":
            candidates.insert(0, ROOT / "hparams/Transformer-Patcher" / f"qwen2vl_{task.lower()}_tpatch_asam_initial_w025.yaml")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if model == "llavaov" and task == "VQA":
            candidates = sorted((ROOT / "hparams/Transformer-Patcher").glob("llavaov_vqa_tpatch_asam_*.yaml"))
            if candidates:
                return candidates[-1]
        raise FileNotFoundError(f"No default T-Patcher ASAM config for {model}/{task}")
    if method == "unike":
        if model == "blip2" and task == "IC" and mode == "baseline":
            return ROOT / "hparams/UniKE/blip2_ic_unike_online_simplified.yaml"
        suffix = "baseline" if mode == "baseline" else "asam"
        return ROOT / "hparams" / "UniKE" / f"{model}_{task.lower()}_unike_{suffix}.yaml"
    if method == "mend":
        return ROOT / "hparams" / "TRAINING" / "MEND" / f"{model}-7b.yaml"
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


def mend_script(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    model_name = {"qwen2vl": "qwen2vl", "llavaov": "llavaov"}[args.model]
    task = args.task.lower()
    if args.phase == "train":
        script = ROOT / f"run_mend_{model_name}_{task}_train.py"
        env_name = "PMRL_MEND_HPARAMS"
    else:
        # Existing evaluation scripts are deliberately reused instead of edited.
        if model_name == "qwen2vl" and task == "ic":
            script = ROOT / ("run_mend_qwen2vl_ic_asam_eval.py" if args.mode == "asam" else "run_mend_qwen2vl_ic_eval.py")
        elif model_name == "qwen2vl" and task == "vqa":
            script = ROOT / "run_mend_qwen2vl_vqa_eval.py"
        else:
            script = ROOT / f"run_mend_{model_name}_{task}_eval.py"
        env_name = "PMRL_MEND_HPARAMS"
    if not script.exists():
        raise FileNotFoundError(f"Existing MEND entrypoint not found: {script}")
    env = os.environ.copy()
    if args.config:
        env[env_name] = str(args.config)
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
    source = {"rewrite_acc": "rewrite_acc", "rephrase_acc": "rephrase_acc", "rephrase_image_acc": "image_rephrase_acc", "locality_acc": "locality_acc", "multimodal_locality_acc": "multimodal_locality_acc"}
    aggregate = {key: mean(scalar(row["post"][value]) for row in metrics) for key, value in source.items()}
    aggregate["gen_avg"] = mean([aggregate["rephrase_acc"], aggregate["rephrase_image_acc"]])
    aggregate["loc_avg"] = mean([aggregate["locality_acc"], aggregate["multimodal_locality_acc"]])
    result = {"repository": "PMRL_MLLM", "method": args.method, "model": args.model, "task": args.task, "mode": args.mode, "config": str(config), "dataset": str(dataset_path), "sample_count": args.size, "seed": args.seed, "wall_seconds": time.time() - started, "metrics": aggregate}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT=" + json.dumps(result))
    return result


def main() -> int:
    args = build_parser().parse_args()
    if args.method == "mend" and args.model == "blip2":
        raise SystemExit("MEND currently has no unified BLIP2 runner; use the existing MEND BLIP2 training config directly.")
    config = args.config or default_config(args.method, args.model, args.task, args.mode)
    if not config.exists() and args.method != "mend":
        raise FileNotFoundError(f"Config does not exist: {config}")
    if args.method == "mend":
        command, env = mend_script(args)
        print(json.dumps({"command": command, "config": str(config) if args.config else None}, indent=2))
        if args.dry_run:
            return 0
        return subprocess.call(command, cwd=ROOT, env=env)
    print(json.dumps({"method": args.method, "model": args.model, "task": args.task, "mode": args.mode, "config": str(config)}, indent=2))
    if args.dry_run:
        return 0
    seed_everything(args.seed)
    run_edit(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
