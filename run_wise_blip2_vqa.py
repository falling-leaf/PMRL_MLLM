"""Run BLIP2 WISE baseline or LAP+PMRL on MMEdit VQA."""
import json, os, time
from pathlib import Path
from statistics import mean
import torch
from easyeditor import CaptionDataset, MultimodalEditor, WISEMultimodalHyperParams

def scalar(v):
    return float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)

def main():
    config_path = os.environ.get("PMRL_WISE_HPARAMS", "hparams/WISE/blip2_ic_baseline.yaml")
    output_dir = Path(os.environ.get("PMRL_OUTPUT_DIR", "results/WISE_VQA_BLIP2"))
    size = int(os.environ.get("PMRL_TEST_SIZE", "100"))
    output_dir.mkdir(parents=True, exist_ok=True)
    hparams = WISEMultimodalHyperParams.from_hparams(config_path)
    mode = "lap_pmrl" if hparams.using_extra else "baseline"
    dataset = CaptionDataset("/root/MMEdit/editing-data/vqa/vqa_eval.json", config=hparams, size=size)
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, _, _ = editor.edit_dataset(ds=dataset, keep_original_weight=True, verbose=True)
    src_keys = {"rewrite_acc":"rewrite_acc","rephrase_acc":"rephrase_acc","rephrase_image_acc":"image_rephrase_acc","locality_acc":"locality_acc","multimodal_locality_acc":"multimodal_locality_acc"}
    agg = {ok: mean(scalar(r["post"][sk]) for r in metrics) for ok, sk in src_keys.items()}
    agg["gen_avg"] = mean([agg["rephrase_acc"], agg["rephrase_image_acc"]])
    agg["loc_avg"] = mean([agg["locality_acc"], agg["multimodal_locality_acc"]])
    result = {"repository":"PMRL_MLLM","model":"BLIP2 OPT-2.7B","method":"WISE","mode":mode,"using_extra":hparams.using_extra,"using_lap":hparams.using_lap,"using_pmrl":hparams.using_pmrl,"dataset":"MMEdit VQA","sample_count":len(metrics),"metrics":agg}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print("FINAL_RESULT=" + json.dumps(result | {"per_case": "saved to result.json"}))

if __name__ == "__main__":
    main()