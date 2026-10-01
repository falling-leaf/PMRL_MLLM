"""Run Qwen2-VL MEND IC inference using trained weights.
Passes file paths (not PIL images) to the MultimodalEditor."""

import os
import json
from pathlib import Path

import torch

from easyeditor import MultimodalEditor, CaptionDataset
from easyeditor.models.mend.mend_multimodal_hparams import MENDMultimodalHparams


def main():
    config_path = os.environ.get(
        "PMRL_MEND_INFER_HPARAMS", "hparams/MEND/qwen2vl_ic_baseline.yaml"
    )
    eval_size = int(os.environ.get("PMRL_MEND_EVAL_SIZE", "100"))

    hparams = MENDMultimodalHparams.from_hparams(config_path)
    hparams.eval_only = True
    hparams.save = False
    hparams.sequential_edit = True
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)

    # Build the editor (loads model + MEND weights)
    editor = MultimodalEditor.from_hparams(hparams)

    # Load annotation file to get file paths
    with open("/root/MMEdit/editing-data/caption/caption_eval_edit.json") as f:
        annotation = json.load(f)
    if eval_size is not None:
        annotation = annotation[:eval_size]

    vis_root = hparams.coco_image
    rephrase_root = hparams.rephrase_image

    all_metrics = []
    for i, record in enumerate(annotation):
        # Build file paths
        image_path = os.path.join(vis_root, record["image"])
        rephrase_image_path = os.path.join(rephrase_root, record["image_rephrase"])
        locality_image_path = os.path.join(vis_root, record["m_loc"])

        prompts = [record["src"]]
        targets = [record["alt"]]
        image = [image_path]
        rephrase_prompts = [record["rephrase"]]
        rephrase_image = [rephrase_image_path]
        locality_inputs = {
            "text": {
                "prompt": [record["loc"]],
                "ground_truth": [record["loc_ans"]],
            },
            "vision": {
                "image": [locality_image_path],
                "prompt": [record["m_loc_q"]],
                "ground_truth": [record["m_loc_a"]],
            },
        }

        result = editor.edit(
            prompts=prompts,
            targets=targets,
            image=image,
            file_type=["image"],
            rephrase_prompts=rephrase_prompts,
            rephrase_image=rephrase_image,
            locality_inputs=locality_inputs,
            keep_original_weight=False,
            verbose=False,
        )
        # editor.edit() returns (all_metrics_list, edited_model, weights_copy)
        metrics_list = result[0] if isinstance(result, (tuple, list)) and len(result) >= 1 else result
        if isinstance(metrics_list, list) and len(metrics_list) > 0:
            entry = metrics_list[0]
        else:
            entry = metrics_list
        all_metrics.append(entry)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"MEND_IC_INFER progress {i+1}/{len(annotation)}", flush=True)

    # Aggregate metrics
    agg = {}
    for key in ["rewrite_acc", "rephrase_acc", "image_rephrase_acc",
                "locality_acc", "multimodal_locality_acc"]:
        vals = []
        for entry in all_metrics:
            if isinstance(entry, dict):
                post = entry.get("post", {})
                if key in post:
                    val = post[key]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    vals.append(val)
        if vals:
            agg[key] = sum(vals) / len(vals)

    agg["sample_count"] = len(all_metrics)
    agg["gen_avg"] = (agg.get("rephrase_acc", 0) + agg.get("image_rephrase_acc", 0)) / 2
    agg["loc_avg"] = (agg.get("locality_acc", 0) + agg.get("multimodal_locality_acc", 0)) / 2

    print(f"\nMEND_IC_INFERENCE_DONE", flush=True)
    for k, v in agg.items():
        print(f"  {k}: {v:.6f}", flush=True)

    # Save results - use a simple serializable format
    out_dir = Path(hparams.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def tensor_to_val(v):
        if isinstance(v, torch.Tensor):
            return v.item()
        if isinstance(v, dict):
            return {k: tensor_to_val(v) for k, v in v.items()}
        if isinstance(v, (list, tuple)):
            return [tensor_to_val(x) for x in v]
        if hasattr(v, 'tolist'):
            return v.tolist()
        return v
    
    serializable = {
        "metrics": {k: float(v) if not isinstance(v, (int, float)) else v for k, v in agg.items()},
        "per_case": tensor_to_val(all_metrics),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(serializable, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"Results saved to {out_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()