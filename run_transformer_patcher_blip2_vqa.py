"""Run Transformer-Patcher baseline or LAP+PMRL on first N MMEdit VQA records with BLIP-2."""

import json
import os
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from easyeditor import MultimodalEditor, VQADataset
from easyeditor.models.transformer_patcher import TransformerPatcherMultimodalHyperParams


def scalar(value):
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


def main():
    config_path = os.environ.get(
        "TPATCH_HPARAMS", "hparams/Transformer-Patcher/blip2_vqa_tpatch.yaml"
    )
    output_dir = Path(os.environ.get("TPATCH_OUTPUT_DIR", "results/TPATCH_BLIP2_VQA_N100"))
    size = int(os.environ.get("TPATCH_TEST_SIZE", "100"))
    seed = int(os.environ.get("TPATCH_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    hparams = TransformerPatcherMultimodalHyperParams.from_hparams(config_path)
    if hparams.model_name != "blip2":
        raise ValueError("This bounded implementation is validated for BLIP-2 only")
    flags = (hparams.using_extra, hparams.using_lap, hparams.using_pmrl)
    if flags not in ((False, False, False), (True, True, True)):
        raise ValueError(f"Invalid Transformer-Patcher mode flags: {flags}")
    dataset = VQADataset(
        "/root/MMEdit/editing-data/vqa/vqa_eval.json", config=hparams, size=size
    )
    if len(dataset) != size:
        raise RuntimeError(f"Expected {size} effective VQA records, got {len(dataset)}")
    editor = MultimodalEditor.from_hparams(hparams)
    started = time.time()
    metrics, _, _ = editor.edit_dataset(ds=dataset, keep_original_weight=True, verbose=True)
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
        key: mean(scalar(row["post"][source]) for row in metrics)
        for key, source in source_keys.items()
    }
    aggregate["gen_avg"] = mean([aggregate["rephrase_acc"], aggregate["rephrase_image_acc"]])
    aggregate["loc_avg"] = mean([aggregate["locality_acc"], aggregate["multimodal_locality_acc"]])
    per_case = [
        {
            "case_id": row["case_id"],
            "time": scalar(row["time"]),
            "post": {key: scalar(row["post"][source]) for key, source in source_keys.items()},
        }
        for row in metrics
    ]
    result = {
        "repository": "PMRL_MLLM",
        "model": "BLIP2 OPT-2.7B",
        "method": "Transformer-Patcher",
        "paper": "Transformer-Patcher: One Mistake worth One Neuron (ICLR 2023)",
        "paper_url": "https://arxiv.org/abs/2301.09785",
        "official_code_revision": "93ff458",
        "task": "VQA",
        "config_path": config_path,
        "dataset": "MMEdit VQA vqa_eval.json",
        "selection": f"first {size} effective records",
        "seed": seed,
        "sample_count": size,
        "hyperparameters": {
            "add_neuron_num": hparams.add_neuron_num,
            "edit_lr": hparams.edit_lr,
            "n_iter": hparams.n_iter,
            "locality_weight": hparams.locality_weight,
            "inner_params": hparams.inner_params,
            "using_extra": hparams.using_extra,
            "using_lap": hparams.using_lap,
            "using_pmrl": hparams.using_pmrl,
            "using_image_embedding": hparams.using_image_embedding,
            "num_rephrase": hparams.num_rephrase,
            "lap_epsilon": hparams.lap_epsilon,
            "pmrl_tau_alignment": hparams.pmrl_tau_alignment,
            "pmrl_tau_regularization": hparams.pmrl_tau_regularization,
            "pmrl_alignment_weight": hparams.pmrl_alignment_weight,
            "pmrl_regularization_weight": hparams.pmrl_regularization_weight,
            "pmrl_scale": hparams.pmrl_scale,
            "lar_random_start": hparams.lar_random_start,
            "lar_pgd_steps": hparams.lar_pgd_steps,
            "lar_step_size": hparams.lar_step_size,
            "lar_target_loss_weight": hparams.lar_target_loss_weight,
            "export_lap_samples": hparams.export_lap_samples,
        },
        "mode": "lap_pmrl" if hparams.using_extra else "baseline",
        "using_extra": hparams.using_extra,
        "using_lap": hparams.using_lap,
        "using_pmrl": hparams.using_pmrl,
        "wall_seconds": wall_seconds,
        "metrics": aggregate,
        "per_case": per_case,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "metrics.txt").write_text(
        "\n".join([f"mode: {result['mode']}", f"sample_count: {size}", f"wall_seconds: {wall_seconds}"] + [
            f"{key}: {value}" for key, value in aggregate.items()
        ]) + "\n",
        encoding="utf-8",
    )
    print("FINAL_RESULT=" + json.dumps(result | {"per_case": "saved to result.json"}))


if __name__ == "__main__":
    main()
