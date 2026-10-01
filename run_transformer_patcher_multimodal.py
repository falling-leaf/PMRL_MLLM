"""Run Transformer-Patcher baseline or ASAM on first N MMEdit IC/VQA records."""

import json
import os
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from easyeditor import CaptionDataset, MultimodalEditor, VQADataset
from easyeditor.models.transformer_patcher import TransformerPatcherMultimodalHyperParams


def scalar(value):
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


def main():
    config_path = os.environ["TPATCH_HPARAMS"]
    output_dir = Path(os.environ["TPATCH_OUTPUT_DIR"])
    size = int(os.environ.get("TPATCH_TEST_SIZE", "100"))
    seed = int(os.environ.get("TPATCH_SEED", "42"))
    task = os.environ.get("TPATCH_TASK", "IC").upper()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = TransformerPatcherMultimodalHyperParams.from_hparams(config_path)
    flags = (hparams.using_extra, hparams.using_lap, hparams.using_pmrl)
    if flags not in ((False, False, False), (True, True, True)):
        raise ValueError(f"Invalid Transformer-Patcher mode flags: {flags}")
    if hparams.model_name not in {"blip2", "minigpt4"} and not (
        "qwen2-vl" in hparams.model_name.lower() or "llava-onevision" in hparams.model_name.lower()
    ):
        raise ValueError(f"Unsupported Transformer-Patcher model: {hparams.model_name}")
    dataset_cls, dataset_path = (
        (CaptionDataset, "/root/MMEdit/editing-data/caption/caption_eval_edit.json")
        if task == "IC"
        else (VQADataset, "/root/MMEdit/editing-data/vqa/vqa_eval.json")
    )
    dataset = dataset_cls(dataset_path, config=hparams, size=size)
    if len(dataset) != size:
        raise RuntimeError(f"Expected {size} effective {task} records, got {len(dataset)}")
    editor = MultimodalEditor.from_hparams(hparams)
    started = time.time()
    metrics, _, _ = editor.edit_dataset(ds=dataset, keep_original_weight=True, verbose=True)
    wall_seconds = time.time() - started
    if len(metrics) != size:
        raise RuntimeError(f"Expected {size} metric records, got {len(metrics)}")

    source_keys = {
        "rewrite_acc": "rewrite_acc", "rephrase_acc": "rephrase_acc",
        "rephrase_image_acc": "image_rephrase_acc", "locality_acc": "locality_acc",
        "multimodal_locality_acc": "multimodal_locality_acc",
    }
    aggregate = {key: mean(scalar(row["post"][source]) for row in metrics) for key, source in source_keys.items()}
    aggregate["gen_avg"] = mean([aggregate["rephrase_acc"], aggregate["rephrase_image_acc"]])
    aggregate["loc_avg"] = mean([aggregate["locality_acc"], aggregate["multimodal_locality_acc"]])
    per_case = [{"case_id": row["case_id"], "time": scalar(row["time"]),
                 "post": {key: scalar(row["post"][source]) for key, source in source_keys.items()}}
                for row in metrics]
    result = {
        "repository": "PMRL_MLLM", "model": hparams.model_name, "method": "Transformer-Patcher",
        "task": task, "config_path": config_path, "dataset": dataset_path,
        "selection": f"first {size} effective records", "seed": seed, "sample_count": size,
        "mode": "asam" if hparams.using_extra else "baseline", "using_extra": hparams.using_extra,
        "using_lap": hparams.using_lap, "using_pmrl": hparams.using_pmrl,
        "hyperparameters": {key: getattr(hparams, key) for key in (
            "add_neuron_num", "edit_lr", "n_iter", "locality_weight", "inner_params", "num_rephrase",
            "lap_epsilon", "pmrl_scale", "lar_random_start", "lar_pgd_steps", "lar_step_size",
            "lar_target_loss_weight", "adam_eps", "export_lap_samples")},
        "wall_seconds": wall_seconds, "metrics": aggregate, "per_case": per_case,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT=" + json.dumps(result | {"per_case": "saved to result.json"}))


if __name__ == "__main__":
    main()
