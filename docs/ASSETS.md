# Models and data

Weights and the MMEdit images are **not** stored in this repository. A full
MMEdit image dump plus the four MLLM checkpoints is tens of gigabytes; commit
only code and the small JSON edit files if you vendor data at all.

Set two environment variables before any run. Every YAML path that used to
point at `/root/hugging_cache` or `/root/MMEdit` is rewritten onto these roots
when the hyperparameter class loads it.

```bash
export PMRL_MODEL_DIR=/path/to/hugging_cache   # default: ../hugging_cache
export PMRL_DATA_DIR=/path/to/MMEdit           # default: ../MMEdit
```

## Benchmark

MMEdit (Cheng et al., 2023), the Editing-VQA and Editing-IC splits used by the paper.

- Project: https://github.com/zjunlp/EasyEdit
- Data release used by MMEdit: the `editing-data/` JSON plus COCO `val2014` and the rephrased-image folder `val2014_image_rephrase`.
- Expected layout under `PMRL_DATA_DIR`:

```text
editing-data/caption/caption_train_edit.json
editing-data/caption/caption_eval_edit.json
editing-data/vqa/vqa_train.json
editing-data/vqa/vqa_eval.json
editing-data/locality/
editing-data/multimodal_locality/
images/val2014/
images/val2014_image_rephrase/
```

The JSON files are small. The two image folders are not; download COCO val2014 from the COCO site (https://cocodataset.org/#download) and follow the MMEdit README for the rephrased images. Do not copy either folder into this repo.

## Checkpoints

Place each download at the filename below inside `PMRL_MODEL_DIR`. Licenses stay with the original authors.

| File or directory | Used by | Where to get it |
|---|---|---|
| `opt-2.7b/` | BLIP2 OPT-2.7B decoder | https://huggingface.co/facebook/opt-2.7b |
| `opt-125m/` | SERAC counterfactual model | https://huggingface.co/facebook/opt-125m |
| `bert-base-uncased/` | BLIP2 / MiniGPT-4 Q-Former tokenizer | https://huggingface.co/bert-base-uncased |
| `blip2_pretrained_opt2.7b.pth` | BLIP2 Q-Former | https://github.com/salesforce/LAVIS (BLIP-2 OPT-2.7B checkpoint) |
| `blip2_pretrained_flant5xxl.pth` | MiniGPT-4 Q-Former init | https://github.com/salesforce/LAVIS (BLIP-2 FlanT5-XXL checkpoint) |
| `eva_vit_g.pth` | BLIP2 and MiniGPT-4 vision tower | https://github.com/baaivision/EVA (EVA-CLIP ViT-G) |
| `pretrained_minigpt4_7b.pth` | MiniGPT-4 projection | https://github.com/Vision-CAIR/MiniGPT-4 (7B aligned checkpoint) |
| `vicuna-7b-v1.5/` | MiniGPT-4 language model | https://huggingface.co/lmsys/vicuna-7b-v1.5 |
| `Qwen2-VL-7B-Instruct/` | Qwen2-VL backbone | https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct |
| `Qwen2-0.5B-Instruct/` | SERAC counterfactual model for Qwen2-VL | https://huggingface.co/Qwen/Qwen2-0.5B-Instruct |
| `llava-onevision-qwen2-7b-ov-hf/` | LLaVA-OneVision backbone | https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf |
| `all-MiniLM-L6-v2/` | IKE sentence encoder | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| `distilbert-base-cased/` | SERAC scope classifier | https://huggingface.co/distilbert-base-cased |

`huggingface-cli download <repo> --local-dir $PMRL_MODEL_DIR/<name>` is enough for the Hub rows. The three `.pth` files come from the LAVIS / EVA / MiniGPT-4 release pages linked above; keep the filenames in the table.

## What not to commit

`results/`, `run_logs/`, `logs/`, checkpoints, COCO images, and `*.pth` are gitignored. On the machine this repo was prepared on, previous experiment outputs live outside the tree at `/root/autodl-tmp/PMRL_MLLM_artifacts/` and are not part of the paper artifact.
