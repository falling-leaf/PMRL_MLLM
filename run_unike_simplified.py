"""Run online-simplified UniKE on MiniGPT-4 or LLaVA-OneVision."""
import json
import os
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from easyeditor import CaptionDataset, MultimodalEditor, VQADataset
from easyeditor.models.unike_simplified import UniKESimplifiedHyperParams


def scalar(value):
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


def main():
    config_path = os.environ["UNIKE_SIMPLIFIED_HPARAMS"]
    task = os.environ.get("UNIKE_TASK", "IC").upper()
    size = int(os.environ.get("UNIKE_TEST_SIZE", "100"))
    output_dir = Path(os.environ["UNIKE_OUTPUT_DIR"])
    seed = int(os.environ.get("UNIKE_SEED", "42"))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    hp = UniKESimplifiedHyperParams.from_hparams(config_path)
    if hp.model_name == "minigpt4":
        dataset_path = "/root/MMEdit/editing-data/caption/caption_eval_edit.json" if task == "IC" else "/root/MMEdit/editing-data/vqa/vqa_eval.json"
        dataset = CaptionDataset(dataset_path, config=hp, size=size) if task == "IC" else VQADataset(dataset_path, config=hp, size=size)
    elif any(model_id in hp.model_name.lower() for model_id in ("llava-onevision", "qwen2-vl")):
        dataset_path = "/root/MMEdit/editing-data/caption/caption_eval_edit.json" if task == "IC" else "/root/MMEdit/editing-data/vqa/vqa_eval.json"
        dataset = CaptionDataset(dataset_path, config=hp, size=size) if task == "IC" else VQADataset(dataset_path, config=hp, size=size)
    else:
        raise ValueError(f"Unsupported model_name: {hp.model_name}")
    if len(dataset) != size: raise RuntimeError(f"Expected {size} effective records, got {len(dataset)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    editor = MultimodalEditor.from_hparams(hp)
    started = time.time()
    metrics, _, _ = editor.edit_dataset(ds=dataset, keep_original_weight=True, verbose=True)
    wall = time.time() - started
    keys = {"rewrite_acc":"rewrite_acc", "rephrase_acc":"rephrase_acc", "rephrase_image_acc":"image_rephrase_acc", "locality_acc":"locality_acc", "multimodal_locality_acc":"multimodal_locality_acc"}
    aggregate = {k: mean(scalar(row["post"][v]) for row in metrics) for k,v in keys.items()}
    aggregate["gen_avg"] = mean([aggregate["rephrase_acc"], aggregate["rephrase_image_acc"]])
    aggregate["loc_avg"] = mean([aggregate["locality_acc"], aggregate["multimodal_locality_acc"]])
    per_case = [{"case_id": row["case_id"], "time": scalar(row["time"]), "post": {k: scalar(row["post"][v]) for k,v in keys.items()}} for row in metrics]
    result = {"repository":"PMRL_MLLM", "model":hp.model_name, "method":"UniKE-Simplified", "implementation":"online_simplified", "paper_equivalent":False, "task":task, "config_path":config_path, "dataset":f"MMEdit {task} {Path(dataset_path).name}", "selection":f"first {size} effective records", "seed":seed, "sample_count":size, "hyperparameters":{k:getattr(hp,k) for k in ("l_ike_layers","edit_lr","n_iter","locality_weight","feature_shift_scale","using_asam","asam_epsilon","asam_weight")}, "wall_seconds":wall, "metrics":aggregate, "per_case":per_case}
    (output_dir/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir/"metrics.txt").write_text("\n".join([f"{k}: {v}" for k,v in aggregate.items()])+"\n", encoding="utf-8")
    print("FINAL_RESULT="+json.dumps(result|{"per_case":"saved to result.json"}))

if __name__ == "__main__": main()
