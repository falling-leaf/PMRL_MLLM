# Paper matrix

`python run_experiment.py --dry-run` resolves every cell below. `--mode asam` is the ASAM / LAP+PMRL editor; `--mode baseline` is the unmodified editor. `preedit`, `ft`, `ike`, and `serac` have no ASAM mode.

Configs are the canonical YAML the launcher selects. Older tuning files stay in the same directory and are used only with `--config`.

## Loss-based editors (baseline and ASAM, VQA and IC)

| Model | UniKE | T-Patcher | MEND | WISE |
|---|---|---|---|---|
| BLIP2-OPT | `hparams/UniKE/blip2_{task}_unike_{mode}.yaml` | `hparams/Transformer-Patcher/blip2_{task}_tpatch*.yaml` | `hparams/TRAINING/MEND/blip2.yaml` | `hparams/WISE/blip2_{task}_{mode}.yaml` |
| MiniGPT-4 | `hparams/UniKE/minigpt4_{task}_unike_{mode}.yaml` | `hparams/Transformer-Patcher/minigpt4_{task}_tpatch*.yaml` | `hparams/TRAINING/MEND/minigpt4.yaml` | `hparams/WISE/minigpt4_{task}_{mode}.yaml` |
| Qwen2-VL | `hparams/UniKE/qwen2vl_{task}_unike_{mode}.yaml` | `hparams/Transformer-Patcher/qwen2vl_{task}_tpatch*.yaml` | `run_mend_qwen2vl_{task}_{train,eval}.py` | `hparams/WISE/qwen2vl_{task}_{mode}.yaml` |
| LLaVA-OneVision | `hparams/UniKE/llavaov_{task}_unike_{mode}.yaml` | `hparams/Transformer-Patcher/llavaov_{task}_tpatch*.yaml` | `run_mend_llavaov_{task}_{train,eval}.py` | `hparams/WISE/llavaov_{task}_{mode}.yaml` |

`{task}` is `vqa` or `ic`. For UniKE and WISE, `{mode}` is `baseline` or `asam` / `lap_pmrl`. T-Patcher baselines are `*_tpatch.yaml` or `*_tpatch_baseline.yaml`; ASAM is `*_tpatch_asam_w025.yaml` (Qwen2-VL uses `*_tpatch_asam_initial_w025.yaml`).

BLIP-2 UniKE loads `UniKE-BLIP2`. MiniGPT-4, Qwen2-VL, and LLaVA-OneVision load `UniKE-Simplified`. MEND for BLIP-2 and MiniGPT-4 is the training YAML only; Qwen2-VL and LLaVA-OneVision also have `run_mend_*` train and eval scripts. MEND ASAM for the 7B models is `hparams/TRAINING/MEND/{qwen2vl,llavaov}-7b-lap-pmrl.yaml`.

## Reference baselines (no ASAM)

| Model | preedit | FT | IKE | SERAC |
|---|---|---|---|---|
| BLIP2-OPT | `hparams/preedit/blip2.yaml` | `hparams/FT/blip2.yaml` | `hparams/IKE/blip2.yaml` | `hparams/SERAC/blip2.yaml` |
| MiniGPT-4 | `hparams/preedit/minigpt4.yaml` | `hparams/FT/minigpt4.yaml` | `hparams/IKE/minigpt4.yaml` | `hparams/SERAC/minigpt4.yaml` |
| Qwen2-VL | `hparams/preedit/qwen2vl_7b.yaml` | `hparams/FT/qwen2vl_7b.yaml` | `hparams/IKE/qwen2vl_7b.yaml` | `hparams/SERAC/qwen2vl.yaml` |
| LLaVA-OneVision | `hparams/preedit/llavaov_7b.yaml` | `hparams/FT/llavaov_7b.yaml` | `hparams/IKE/llavaov_7b.yaml` | `hparams/SERAC/llavaov.yaml` |

SERAC training YAMLs are `hparams/TRAINING/SERAC/{model}.yaml`, started with `--method serac --phase train`.
