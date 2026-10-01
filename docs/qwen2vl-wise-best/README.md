#Qwen2-VL + WISE：IC / VQA 最优实验启动手册

本文档对应仓库 `/root/PMRL_MLLM` 当前已经真实验证过的代码与配置。

## 1. 最优配置结论

### IC（推荐正式配置）

- 配置：`hparams/WISE/qwen2vl_lar_target/lar-target-w0p5.yaml`
- 推荐权重：`lar_target_loss_weight=0.5`
- N=1000 已验证：Gen-M `93.61% -> 94.78%`，提升 `1.17` 个百分点。
- 结果：`results/QWEN_WISE_IC_FULL/summary.json`

### VQA（按目标选择）

- 追求 N=100 Gen-M：`hparams/WISE/qwen2vl_vqa_tuning/target_w0p25.yaml`
  - Gen-M `95.32% -> 96.69%`，提升 `1.37` 个百分点。
  - Loc-T 下降 `4.35` 个百分点。
- 更重视 locality 折中：使用同一配置，但将 `lar_target_loss_weight` 改为 `0.10`。
  - Gen-M 提升 `0.83` 个百分点；Loc-T 下降 `1.76` 个百分点。
- 注意：VQA N=1000 上 weight=0.5 的 Gen-M 提升不显著，且 Loc-T 明显下降，因此不要把 IC 的 weight=0.5 当作 VQA 默认最优。

## 2. 环境与资产

```text
仓库: /root/PMRL_MLLM
Python: /root/miniconda3/envs/easyedit/bin/python
模型: /root/hugging_cache/Qwen2-VL-7B-Instruct
GPU: CUDA device 0，建议约 24 GB 以上显存
IC 数据: /root/MMEdit/editing-data/caption/caption_eval_edit.json
VQA 数据: /root/MMEdit/editing-data/vqa/vqa_eval.json
图像: /root/MMEdit/images
```

进入仓库：

```bash
cd /root/PMRL_MLLM
```

运行前检查：

```bash
nvidia-smi
test -d /root/hugging_cache/Qwen2-VL-7B-Instruct
test -f /root/MMEdit/editing-data/caption/caption_eval_edit.json
test -f /root/MMEdit/editing-data/vqa/vqa_eval.json
test -d /root/MMEdit/images
```

所有实验离线运行：

```bash
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/root/PMRL_MLLM
```

## 3. 通用入口参数

入口：

```text
run_wise_qwen2vl_ic.py
```

虽然文件名包含 `ic`，但它通过 `PMRL_TASK` 同时支持 IC 和 VQA。

环境变量：

| 变量 | 含义 | 示例 |
|---|---|---|
| `PMRL_TASK` | 任务 | `IC` / `VQA` |
| `PMRL_TEST_SIZE` | 前 N 条有效数据 | `100` / `1000` |
| `PMRL_SEED` | 随机种子 | `42` |
| `PMRL_WISE_HPARAMS` | YAML 配置路径 | 见下文 |
| `PMRL_OUTPUT_DIR` | 独立输出目录 | `results/...` |

每次运行输出：

```text
result.json   # 汇总 + 每 case 指标
metrics.txt   # 汇总指标
stdout 中一个 FINAL_RESULT
```

## 4. IC 启动方式

### 4.1 IC 最优增强 N=100

```bash
cd /root/PMRL_MLLM
PMRL_TASK=IC \
PMRL_TEST_SIZE=100 \
PMRL_SEED=42 \
PMRL_WISE_HPARAMS=hparams/WISE/qwen2vl_lar_target/lar-target-w0p5.yaml \
PMRL_OUTPUT_DIR=results/MANUAL_QWEN_WISE_IC_BEST_N100 \
/root/miniconda3/envs/easyedit/bin/python run_wise_qwen2vl_ic.py \
  > run_ic_best_n100.log 2>&1
```

### 4.2 IC baseline N=100

```bash
PMRL_TASK=IC \
PMRL_TEST_SIZE=100 \
PMRL_SEED=42 \
PMRL_WISE_HPARAMS=hparams/WISE/qwen2vl_ic_baseline.yaml \
PMRL_OUTPUT_DIR=results/MANUAL_QWEN_WISE_IC_BASELINE_N100 \
/root/miniconda3/envs/easyedit/bin/python run_wise_qwen2vl_ic.py \
  > run_ic_baseline_n100.log 2>&1
```

全量 IC 将 `PMRL_TEST_SIZE` 改为 `1000`，并使用新的输出目录。不要覆盖现有正式结果。

## 5. VQA 启动方式

### 5.1 VQA 当前 Gen-M 最优 N=100（weight=0.25）

```bash
cd /root/PMRL_MLLM
PMRL_TASK=VQA \
PMRL_TEST_SIZE=100 \
PMRL_SEED=42 \
PMRL_WISE_HPARAMS=hparams/WISE/qwen2vl_vqa_tuning/target_w0p25.yaml \
PMRL_OUTPUT_DIR=results/MANUAL_QWEN_WISE_VQA_BEST_W025_N100 \
/root/miniconda3/envs/easyedit/bin/python run_wise_qwen2vl_ic.py \
  > run_vqa_best_w025_n100.log 2>&1
```

### 5.2 VQA locality 折中版 N=100（weight=0.10）

```bash
PMRL_TASK=VQA \
PMRL_TEST_SIZE=100 \
PMRL_SEED=42 \
PMRL_WISE_HPARAMS=hparams/WISE/qwen2vl_vqa_tuning/target_w0p10.yaml \
PMRL_OUTPUT_DIR=results/MANUAL_QWEN_WISE_VQA_BALANCED_W010_N100 \
/root/miniconda3/envs/easyedit/bin/python run_wise_qwen2vl_ic.py \
  > run_vqa_balanced_w010_n100.log 2>&1
```

### 5.3 VQA baseline N=100

```bash
PMRL_TASK=VQA \
PMRL_TEST_SIZE=100 \
PMRL_SEED=42 \
PMRL_WISE_HPARAMS=hparams/WISE/qwen2vl_vqa_baseline.yaml \
PMRL_OUTPUT_DIR=results/MANUAL_QWEN_WISE_VQA_BASELINE_N100 \
/root/miniconda3/envs/easyedit/bin/python run_wise_qwen2vl_ic.py \
  > run_vqa_baseline_n100.log 2>&1
```

## 6. 最优增强超参数

共同参数：

```yaml
model_name: /root/hugging_cache/Qwen2-VL-7B-Instruct
dtype: torch.bfloat16
device: 0
inner_params:
  - model.language_model.layers.23.mlp.down_proj.weight

n_iter: 10
edit_lr: 1.0
mask_ratio: 0.2
norm_constraint: 1.0
act_margin: [5.0, 20.0, 10.0]
act_ratio: 0.4
densities: 0.53
weights: 1.0

using_extra: true
using_lap: true
using_pmrl: true
using_image_embedding: true
num_rephrase: 5
lap_epsilon: 0.001

lar_random_start: true
lar_pgd_steps: 1
lar_step_size: 0.001
lar_joint_perturbation: true

pmrl_tau_alignment: 0.01
pmrl_tau_regularization: 0.1
pmrl_alignment_weight: 1.0
pmrl_regularization_weight: 0.5
pmrl_scale: 0.01
pmrl_spectral_alignment: false
```

任务差异：

```yaml
# IC
lar_target_loss_weight: 0.5

# VQA：Gen-M 优先
lar_target_loss_weight: 0.25

# VQA：locality 折中
lar_target_loss_weight: 0.10
```

## 7. 后台启动

推荐使用仓库提供的包装脚本（见本目录 `scripts/`），它会创建独立日志、状态和输出目录：

```bash
bash docs/qwen2vl-wise-best/scripts/run_best.sh IC 100 ic_best_n100
bash docs/qwen2vl-wise-best/scripts/run_baseline.sh IC 100 ic_baseline_n100
bash docs/qwen2vl-wise-best/scripts/run_best.sh VQA 100 vqa_best_n100
bash docs/qwen2vl-wise-best/scripts/run_baseline.sh VQA 100 vqa_baseline_n100
```

若需让 shell 退出后继续：

```bash
nohup bash docs/qwen2vl-wise-best/scripts/run_best.sh IC 100 ic_best_n100 \
  > /root/PMRL_MLLM/run_logs/manual_ic_launcher.log 2>&1 &
```

查看：

```bash
ps aux | grep run_wise_qwen2vl_ic.py
nvidia-smi
tail -f run_logs/qwen2vl-wise-best/<run-id>/run.log
```

## 8. 完成判定

不能只看进程退出。必须全部满足：

```bash
cat run_logs/qwen2vl-wise-best/<run-id>/status.txt
grep -c 'FINAL_RESULT=' run_logs/qwen2vl-wise-best/<run-id>/run.log
python - <<'PY'
import json
from pathlib import Path
p=Path('results/qwen2vl-wise-best/<run-id>/result.json')
d=json.loads(p.read_text())
print('task=',d['task'])
print('mode=',d['mode'])
print('sample_count=',d['sample_count'])
print('per_case=',len(d['per_case']))
print('metrics=',d['metrics'])
PY
```

预期：

- `EXIT_CODE=0`；
- `FINAL_RESULT` 恰好 1 个；
- `sample_count == len(per_case) == 请求 N`；
- case ID 连续；
- baseline 日志 `loss_alignment:` 为 0；
- 增强日志 `loss_alignment:` 和 `lar_variant_target_loss:` 均为 `N × 10`。

## 9. 指标解释

```text
Acc     = rewrite_acc
Gen-T   = rephrase_acc
Gen-M   = rephrase_image_acc
Gen Avg = (Gen-T + Gen-M) / 2
Loc-T   = locality_acc
Loc-M   = multimodal_locality_acc
Loc Avg = (Loc-T + Loc-M) / 2
```

## 10. 常见问题

1. 不要修改 baseline 来追求增强收益；baseline 配置必须固定。
2. IC 和 VQA 必须设置正确的 `PMRL_TASK`，否则会加载错误数据集。
3. 每次使用新的输出目录，避免覆盖或误用历史结果。
4. VQA 可能出现 target CE 已饱和、输入梯度为 0；当前 randomized LAR 已支持有限零梯度随机起点，但 NaN/Inf 仍会显式失败。
5. VQA weight=0.25 追求 Gen-M，但 locality 有损失；不要把它描述为所有指标全面最优。
6. 先跑 N=1 smoke，再跑 N=100；只有需要正式结论时才跑 N=1000。

## 11. 已验证结果索引

```text
IC 全量: results/QWEN_WISE_IC_FULL/summary.json
IC 注册表: results/QWEN_WISE_IC_EXPERIMENT_REGISTRY.md
IC 调参表: results/QWEN_WISE_IC_PMRL_TUNING.md
VQA N=100: results/QWEN_WISE_VQA_N100/summary.json
VQA N=1000: results/QWEN_WISE_VQA_N1000/summary.json
VQA 限时调参: results/QWEN_WISE_VQA_TUNING_N100/summary.json
VQA 注册表: results/QWEN_WISE_VQA_EXPERIMENT_REGISTRY.md
最终摘要: /root/final.txt
```
