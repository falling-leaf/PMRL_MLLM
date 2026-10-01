# PMRL_MLLM 快速验证命令

标准化命名：`run_{method}_{model}_{task}.py`，通过 `PMRL_*` 环境变量切换。

## WISE（推理型，无需训练）

| 模型 | IC | VQA |
|------|----|-----|
| BLIP2 | `python run_wise_blip2_ic.py` | `python run_wise_blip2_vqa.py` |
| MiniGPT-4 | `python run_wise_minigpt4_ic.py` | — |
| Qwen2-VL | `python run_wise_qwen2vl_ic.py` | `python run_wise_qwen2vl_vqa.py` |
| LLaVA-OV | `python run_wise_llava_ic.py` | `python run_wise_llava_vqa.py` |

**ASAM 切换**（所有 WISE 脚本通用）：
```bash
PMRL_WISE_HPARAMS=hparams/WISE/{model}_{task}_lap_pmrl.yaml python run_wise_{model}_{task}.py
```

## T-Patcher（推理型）

| 模型 | IC | VQA |
|------|----|-----|
| BLIP2 | `python run_transformer_patcher_blip2_ic.py` | `python run_transformer_patcher_blip2_vqa.py` |
| MiniGPT-4 | `run_transformer_patcher_multimodal.py` | `run_transformer_patcher_multimodal.py` |
| Qwen2-VL | `run_transformer_patcher_multimodal.py` | `run_transformer_patcher_multimodal.py` |
| LLaVA-OV | `run_transformer_patcher_multimodal.py` | `run_transformer_patcher_multimodal.py` |

## MEND（需先训练）

**训练：**
```bash
# Qwen2-VL IC baseline
PMRL_MEND_TRAIN_SIZE=1000 PMRL_MEND_VAL_SIZE=100 python run_mend_qwen2vl_ic_train.py

# Qwen2-VL IC LAP+PMRL
PMRL_MEND_HPARAMS=hparams/TRAINING/MEND/qwen2vl-7b-lap-pmrl.yaml \
  PMRL_MEND_TRAIN_SIZE=1000 PMRL_MEND_VAL_SIZE=100 python run_mend_qwen2vl_ic_train.py

# Qwen2-VL VQA baseline
PMRL_MEND_HPARAMS=hparams/TRAINING/MEND/qwen2vl-7b.yaml python run_mend_qwen2vl_vqa_train.py

# Qwen2-VL VQA LAP+PMRL
PMRL_MEND_HPARAMS=hparams/TRAINING/MEND/qwen2vl-7b-lap-pmrl.yaml python run_mend_qwen2vl_vqa_train.py

# LLaVA-OV IC baseline（训练中，~27h）
python run_mend_llavaov_ic_train.py
```

**评估（训练完成后）：**
```bash
# Qwen2-VL IC baseline eval
python run_mend_qwen2vl_ic_eval.py

# Qwen2-VL IC ASAM eval
python run_mend_qwen2vl_ic_asam_eval.py

# Qwen2-VL VQA eval（自动 baseline + ASAM）
python run_mend_qwen2vl_vqa_eval.py

# LLaVA-OV eval
python run_mend_llavaov_ic_eval.py
python run_mend_llavaov_vqa_eval.py
```

## UniKE（推理型）

| 模型 | IC | VQA |
|------|----|-----|
| BLIP2 | `python run_unike_blip2_ic.py` | `python run_unike_simplified.py` |
| MiniGPT-4 | `python run_unike_simplified.py` | `python run_unike_simplified.py` |
| Qwen2-VL | `python run_unike_simplified.py` | `python run_unike_simplified.py` |
| LLaVA-OV | `python run_unike_simplified.py` | `python run_unike_simplified.py` |

## 通用环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `PMRL_WISE_HPARAMS` | WISE 配置路径 | `hparams/WISE/{model}_{task}_baseline.yaml` |
| `PMRL_MEND_HPARAMS` | MEND 训练配置路径 | `hparams/TRAINING/MEND/{model}-7b.yaml` |
| `PMRL_MEND_EVAL_SIZE` | MEND 评估样本数 | `100` |
| `PMRL_MEND_TRAIN_SIZE` | MEND 训练样本数 | `1000` |
| `PMRL_MEND_VAL_SIZE` | MEND 验证样本数 | `100` |
| `PMRL_MEND_MAX_ITERS` | MEND 最大迭代数 | 配置中的 `max_iters` |
| `PMRL_OUTPUT_DIR` | 输出目录 | `results/{METHOD}_{TASK}_{MODEL}` |
| `PMRL_TEST_SIZE` | 测试样本数 | `100` |