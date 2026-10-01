"""Run isolated UniKE-BLIP2 on the first N MMEdit IC records."""
import json
import os
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from easyeditor import CaptionDataset, MultimodalEditor, VQADataset
from easyeditor.models.unike_blip2 import UniKEBLIP2HyperParams


def scalar(value):
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


def main():
    config_path = os.environ.get("UNIKE_HPARAMS", "hparams/UniKE/blip2_ic_unike_online_simplified.yaml")
    output_dir = Path(os.environ.get("UNIKE_OUTPUT_DIR", "results/UNIKE_BLIP2_IC"))
    size = int(os.environ.get("UNIKE_TEST_SIZE", "100"))
    task = os.environ.get("UNIKE_TASK", "IC").upper()
    seed = int(os.environ.get("UNIKE_SEED", "42"))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    hparams = UniKEBLIP2HyperParams.from_hparams(config_path)
    if task == "IC":
        dataset_path = "/root/MMEdit/editing-data/caption/caption_eval_edit.json"
        dataset = CaptionDataset(dataset_path, config=hparams, size=size)
    elif task == "VQA":
        dataset_path = "/root/MMEdit/editing-data/vqa/vqa_eval.json"
        dataset = VQADataset(dataset_path, config=hparams, size=size)
    else:
        raise ValueError(f"Unsupported UNIKE_TASK={task}; expected IC or VQA")
    if len(dataset) != size: raise RuntimeError(f"Expected {size} effective IC records, got {len(dataset)}")
    editor = MultimodalEditor.from_hparams(hparams)
    started = time.time()
    metrics, _, _ = editor.edit_dataset(ds=dataset, keep_original_weight=True, verbose=True)
    wall_seconds = time.time() - started
    if len(metrics) != size: raise RuntimeError(f"Expected {size} metric records, got {len(metrics)}")
    source = {"rewrite_acc": "rewrite_acc", "rephrase_acc": "rephrase_acc", "rephrase_image_acc": "image_rephrase_acc", "locality_acc": "locality_acc", "multimodal_locality_acc": "multimodal_locality_acc"}
    aggregate = {key: mean(scalar(row["post"][source_key]) for row in metrics) for key, source_key in source.items()}
    aggregate["gen_avg"] = mean([aggregate["rephrase_acc"], aggregate["rephrase_image_acc"]])
    aggregate["loc_avg"] = mean([aggregate["locality_acc"], aggregate["multimodal_locality_acc"]])
    per_case = [{"case_id": row["case_id"], "time": scalar(row["time"]), "post": {key: scalar(row["post"][source_key]) for key, source_key in source.items()}} for row in metrics]
    result = {"repository": "PMRL_MLLM", "model": "BLIP2 OPT-2.7B", "method": "UniKE-BLIP2", "task": task, "implementation": "online_simplified" if not hparams.retrieved_knowledge_path else "official_artifact", "paper_equivalent": bool(hparams.retrieved_knowledge_path and hparams.latent_ike_path and hparams.semantic_encoder_path), "official_code_revision": "dcf267e", "config_path": config_path, "dataset": f"MMEdit {task} {Path(dataset_path).name}", "selection": f"first {size} effective records", "seed": seed, "sample_count": size, "hyperparameters": {"add_neuron_num": hparams.add_neuron_num, "edit_lr": hparams.edit_lr, "n_iter": hparams.n_iter, "locality_weight": hparams.locality_weight, "l_ike_layers": hparams.l_ike_layers, "retrieval_top_k": hparams.retrieval_top_k, "feature_shift_scale": hparams.feature_shift_scale, "semantic_gate": hparams.semantic_gate, "using_asam": hparams.using_asam, "asam_epsilon": hparams.asam_epsilon, "asam_weight": hparams.asam_weight, "asam_visual_only": hparams.asam_visual_only, "adam_eps": hparams.adam_eps}, "wall_seconds": wall_seconds, "metrics": aggregate, "per_case": per_case}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "metrics.txt").write_text("\n".join([f"implementation: {result['implementation']}", f"paper_equivalent: {result['paper_equivalent']}", f"sample_count: {size}", f"wall_seconds: {wall_seconds}"] + [f"{key}: {value}" for key, value in aggregate.items()]) + "\n", encoding="utf-8")
    print("FINAL_RESULT=" + json.dumps(result | {"per_case": "saved to result.json"}))


if __name__ == "__main__":
    main()
