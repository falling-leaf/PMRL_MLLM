# PMRL-MLLM

Code for the multimodal knowledge-editing experiments behind **ASAM** (adversarial semantic alignment) on MMEdit. The repository is a fork of [EasyEdit](https://github.com/zjunlp/EasyEdit) with four MLLM backbones and the editors used in the paper. Model weights and COCO images are **not** in the tree; see [docs/ASSETS.md](docs/ASSETS.md).

## What runs

One entry point covers the paper matrix. `--mode asam` turns on the editor's ASAM / LAP+PMRL objective. `--mode baseline` is the unmodified editor. `preedit`, `serac`, `ike`, and `ft` are reference baselines and reject `--mode asam`.

| Editor | BLIP2-OPT | MiniGPT-4 | Qwen2-VL | LLaVA-OneVision | ASAM |
|---|---|---|---|---|---|
| UniKE | `unike_blip2` | `unike_simplified` | `unike_simplified` | `unike_simplified` | yes |
| T-Patcher | paired FFN neurons | paired FFN neurons | SwiGLU triplet | SwiGLU triplet | yes |
| MEND | training YAML | training YAML | `run_mend_qwen2vl_*` | `run_mend_llavaov_*` | yes |
| WISE | layer-23 FFN | layer-23 down-proj | layer-23 down-proj | layer-23 down-proj | yes |
| preedit / FT / IKE / SERAC | baseline only | baseline only | baseline only | baseline only | no |

Tasks are MMEdit **VQA** and **IC**. The cell-by-cell config map is [docs/MATRIX.md](docs/MATRIX.md). Canonical configs live in `hparams/{WISE,Transformer-Patcher,UniKE,TRAINING/MEND,preedit,FT,IKE,SERAC}/`. Older tuning YAMLs are kept next to them; the launcher never picks a tuning file unless you pass `--config`.

## Setup

```bash
pip install -r requirements.txt
export PMRL_MODEL_DIR=/path/to/hugging_cache
export PMRL_DATA_DIR=/path/to/MMEdit
```

Download links and the expected directory layout are in [docs/ASSETS.md](docs/ASSETS.md). Paths inside YAML files are rewritten onto those two roots at load time, so a checkout does not depend on this machine's `/root` paths.

## Run

Resolve the config without loading a model:

```bash
python run_experiment.py --method wise --model blip2 --task IC --mode baseline --dry-run
python run_experiment.py --method unike --model llavaov --task VQA --mode asam --dry-run
```

Edit and evaluate (N=100 is the paper default; start with `--size 1`):

```bash
python run_experiment.py \
  --method tpatch --model qwen2vl --task VQA --mode asam \
  --size 100 --seed 42
```

The same command works for `--method {wise,tpatch,unike,preedit,ike,ft,serac}` and `--model {blip2,minigpt4,qwen2vl,llavaov}`. Metrics are written to `results/{method}_{model}_{task}_{mode}/result.json` (reliability, text generality, image generality, text locality, multimodal locality).

MEND and SERAC need a training phase before eval:

```bash
# Qwen2-VL or LLaVA-OneVision. This forwards to the existing run_mend_* script.
python run_experiment.py --method mend --model qwen2vl --task IC --mode baseline --phase train

# SERAC counterfactual model. BLIP-2 / MiniGPT-4 / Qwen2-VL / LLaVA-OneVision.
python run_experiment.py --method serac --model blip2 --task VQA --mode baseline --phase train --size 1000
```

BLIP-2 and MiniGPT-4 MEND have training YAMLs (`hparams/TRAINING/MEND/blip2.yaml`, `minigpt4.yaml`) but no `run_mend_*` wrapper. `--dry-run` prints the YAML; train them with `MultimodalTrainer` the same way `run_mend_qwen2vl_ic_train.py` does.

## Layout

```text
run_experiment.py          paper-matrix launcher
easyeditor/models/         WISE, T-Patcher, UniKE, MEND, SERAC, IKE, FT
easyeditor/util/hparams.py path rewrite (PMRL_MODEL_DIR, PMRL_DATA_DIR)
hparams/                   one canonical YAML per paper cell, plus tuning history
docs/ASSETS.md             download links, no weights in git
```

`results/`, `run_logs/`, checkpoints, and `*.pth` are gitignored on purpose.

## Upstream

The text-only editors, the original README, and the EasyEdit tutorial notebooks remain for context. Multimodal editing is the path this repository is maintained for. EasyEdit's license is in `LICENSE`.
