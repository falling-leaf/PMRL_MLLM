"""Run LLaVA-OneVision WISE baseline or LAP+PMRL on MMEdit IC."""

import json
import os
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from easyeditor import CaptionDataset, MultimodalEditor, WISEMultimodalHyperParams


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def main():
    config_path = os.environ.get(
        "PMRL_WISE_HPARAMS", "hparams/WISE/llavaov_ic_baseline.yaml"
    )
    output_dir = Path(os.environ.get("PMRL_OUTPUT_DIR", "results/WISE_IC_LLAVAOV"))
    size = int(os.environ.get("PMRL_TEST_SIZE", "100"))
    seed = int(os.environ.get("PMRL_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = WISEMultimodalHyperParams.from_hparams(config_path)
    if "llava-onevision" not in hparams.model_name.lower():
        raise ValueError(f"LLaVA runner received non-LLaVA model: {hparams.model_name}")
    flags = (hparams.using_extra, hparams.using_lap, hparams.using_pmrl)
    if flags not in ((False, False, False), (True, True, True)):
        raise ValueError(f"Invalid WISE mode flags: {flags}")
    mode = "lap_pmrl" if hparams.using_extra else "baseline"
    dataset_path = "/root/MMEdit/editing-data/caption/caption_eval_edit.json"
    dataset = CaptionDataset(dataset_path, config=hparams, size=size)
    if len(dataset) != size:
        raise RuntimeError(f"Expected {size} effective IC records, got {len(dataset)}")

    editor = MultimodalEditor.from_hparams(hparams)
    started = time.time()
    metrics, _, _ = editor.edit_dataset(
        ds=dataset, keep_original_weight=True, verbose=True
    )
    wall_seconds = time.time() - started
    if len(metrics) != size:
        raise RuntimeError(f"Expected {size} metric records, got {len(metrics)}")

    source_keys = {
        "rewrite_acc": "rewrite_acc",
        "rephrase_acc": "rephrase_acc",
        "rephrase_image_acc": "image_rephrase_acc",
        "locality_acc": "locality_acc",
        "multimodal_locality_acc": "multimodal_locality_acc",
    }
    aggregate = {
        output_key: mean(scalar(row["post"][source_key]) for row in metrics)
        for output_key, source_key in source_keys.items()
    }
    aggregate["gen_avg"] = mean(
        [aggregate["rephrase_acc"], aggregate["rephrase_image_acc"]]
    )
    aggregate["loc_avg"] = mean(
        [aggregate["locality_acc"], aggregate["multimodal_locality_acc"]]
    )
    per_case = [
        {
            "case_id": row["case_id"],
            "time": scalar(row["time"]),
            "post": {
                output_key: scalar(row["post"][source_key])
                for output_key, source_key in source_keys.items()
            },
        }
        for row in metrics
    ]
    result = {
        "repository": "PMRL_MLLM",
        "model": "LLaVA-OneVision-Qwen2-7B-OV-HF",
        "model_path": hparams.model_name,
        "method": "WISE",
        "mode": mode,
        "task": "IC",
        "config_path": config_path,
        "using_extra": hparams.using_extra,
        "using_lap": hparams.using_lap,
        "using_pmrl": hparams.using_pmrl,
        "dataset": "MMEdit IC caption_eval_edit.json",
        "selection": f"first {size} effective records",
        "seed": seed,
        "sample_count": size,
        "wall_seconds": wall_seconds,
        "metrics": aggregate,
        "per_case": per_case,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        f"mode: {mode}",
        f"sample_count: {size}",
        f"wall_seconds: {wall_seconds}",
    ]
    lines.extend(f"{key}: {value}" for key, value in aggregate.items())
    (output_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("FINAL_RESULT=" + json.dumps(result | {"per_case": "saved to result.json"}))


if __name__ == "__main__":
    main()
