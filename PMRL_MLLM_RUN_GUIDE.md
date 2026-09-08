# PMRL_MLLM 统一运行入口

本次整理新增 `run_experiment.py`，将 WISE、T-Patcher、UniKE 的重复 run 脚本统一为一个 argparse 入口；MEND 仅做安全转发，继续调用仓库原有训练/评估脚本，不修改 MEND 实验逻辑，避免影响正在运行的实验。

## 运行前提

```bash
cd /root/PMRL_MLLM
```

需要已经安装仓库依赖，并准备好本地 MMEdit 数据：

- IC: `/root/MMEdit/editing-data/caption/caption_eval_edit.json`
- VQA: `/root/MMEdit/editing-data/vqa/vqa_eval.json`

模型权重和数据下载不由本入口自动执行。YAML 中的模型路径应改为当前机器上的本地路径。

## 统一命令格式

```bash
python run_experiment.py \
  --method {wise,tpatch,unike,mend} \
  --model {blip2,minigpt4,qwen2vl,llavaov} \
  --task {IC,VQA} \
  --mode {baseline,asam} \
  [--config PATH] [--size N] [--seed 42] \
  [--output-dir PATH] [--phase edit|train|eval]
```

先用 `--dry-run` 检查配置解析，不加载模型：

```bash
python run_experiment.py --method wise --model blip2 --task IC --mode baseline --dry-run
```

默认使用 100 个有效样本；正式比较应使用相同的 case 集和相同的 `--size`，推荐先 smoke test，再进行 N=100 对比。

## 四种方法的统一入口

### WISE

WISE 是编辑时执行的模型权重编辑方法。baseline 使用普通 WISE；`--mode asam` 对应仓库命名中的 LAP+PMRL 增强配置（`using_extra=true`、`using_lap=true`、`using_pmrl=true`）。

```bash
python run_experiment.py --method wise --model blip2 --task IC --mode baseline
python run_experiment.py --method wise --model qwen2vl --task VQA --mode asam --size 100
```

支持的模型配置覆盖：BLIP2、MiniGPT-4、Qwen2-VL、LLaVA-OneVision；IC/VQA 配置位于 `hparams/WISE/`。也可以明确指定调参后的 YAML：

```bash
python run_experiment.py --method wise --model llavaov --task VQA --mode asam \
  --config hparams/WISE/llavaov_vqa_tuning_epsilon002_v7.yaml
```

### T-Patcher

T-Patcher 通过新增/修改神经元进行编辑。baseline 配置关闭 `using_extra/using_lap/using_pmrl`；ASAM 配置通常使用 `tpatch_asam_*` 或 `tpatch_pmrl_*` YAML。

```bash
python run_experiment.py --method tpatch --model blip2 --task VQA --mode baseline
python run_experiment.py --method tpatch --model qwen2vl --task IC --mode asam \
  --config hparams/Transformer-Patcher/qwen2vl_ic_tpatch_asam_initial_w025.yaml
```

T-Patcher 的历史调参文件较多，统一入口不会删除它们；正式实验应通过 `--config` 明确指定冻结的 baseline/ASAM 配置，避免误用调参文件。

### UniKE

UniKE 当前代码包含模型专用实现和 simplified/online 实现：

- BLIP2 通过 `easyeditor.models.unike_blip2`；
- MiniGPT-4、Qwen2-VL、LLaVA-OneVision 通过 `easyeditor.models.unike_simplified`；
- `--mode asam` 选择 `using_asam=true` 配置。

```bash
python run_experiment.py --method unike --model blip2 --task IC --mode baseline
python run_experiment.py --method unike --model minigpt4 --task VQA --mode asam \
  --config hparams/UniKE/minigpt4_vqa_unike_simplified_asam.yaml
python run_experiment.py --method unike --model llavaov --task IC --mode baseline
```

BLIP2 IC baseline 默认使用 `blip2_ic_unike_online_simplified.yaml`；若要与 ASAM 做严格配对，建议使用明确的 BLIP2 baseline/ASAM YAML 并同时固定 `--size`、`--seed` 和 case 顺序。

### MEND（保护现有实验）

MEND 不在统一入口中重写训练器或评估器。`run_experiment.py` 只转发到现有 `run_mend_*` 文件，因此当前正在运行的 MEND 实验不会被统一入口替换或中断。

```bash
# 仅解析将要执行的原有脚本
python run_experiment.py --method mend --model qwen2vl --task IC \
  --phase eval --mode baseline --dry-run

# 调用原有训练入口；仍建议在独立终端/后台运行
python run_experiment.py --method mend --model qwen2vl --task IC \
  --phase train --mode baseline --config hparams/TRAINING/MEND/qwen2vl-7b.yaml

# LLaVA-OV 训练同理
python run_experiment.py --method mend --model llavaov --task VQA \
  --phase train --config hparams/TRAINING/MEND/llavaov-7b-vqa.yaml
```

MEND 的现有脚本、配置和训练状态文件没有在本次统一入口中合并、删除或改写。MEND 的具体 checkpoint/eval 参数仍以原脚本和 YAML 为准。

## 方法/模型实现概览

| 方法 | BLIP2 | MiniGPT-4 | Qwen2-VL | LLaVA-OV | baseline/ASAM 入口 |
|---|---|---|---|---|---|
| MEND | 原有 MEND 配置/脚本 | 原有训练配置 | 原有训练与评估脚本 | 原有训练与评估脚本 | 仅转发，不改 MEND 核心 |
| WISE | `wise_multimodal_hparams.py` | 同 WISE 编辑器 | 同 WISE 编辑器 | 同 WISE 编辑器 | `hparams/WISE/*_baseline.yaml` / `*_lap_pmrl.yaml` |
| T-Patcher | `transformer_patcher_multimodal_hparams.py` | 同统一编辑器 | 同统一编辑器 | 同统一编辑器 | `hparams/Transformer-Patcher/` |
| UniKE | `unike_blip2` | `unike_simplified` | `unike_simplified` | `unike_simplified` | `hparams/UniKE/*_baseline.yaml` / `*_asam.yaml` |

这里的 `ASAM` 是本仓库实验配置中的增强模式统称：WISE/T-Patcher 常体现为 LAP+PMRL 开关，UniKE 使用 `using_asam`、`asam_epsilon`、`asam_weight` 等字段。不要仅根据文件名判断，最终以 YAML 中的开关和实际加载的 HyperParams 为准。

## 输出与验证

统一入口会在输出目录写入轻量的 `result.json`，包含方法、模型、任务、配置、样本数、运行时间和聚合指标。默认输出目录为：

```text
results/{method}_{model}_{task}_{mode}/result.json
```

建议验证：

```bash
python run_experiment.py --help
python run_experiment.py --method tpatch --model blip2 --task IC --mode baseline --dry-run
python -m py_compile run_experiment.py
```

不要把 `results/`、`run_logs/`、checkpoint、模型权重、数据集或其他大体积结果文件加入提交。提交时只保留代码入口和本 Markdown 文档，以及必要的小型配置/测试变更。

## 本次整理的边界

- 新增一个 argparse 入口，减少四种方法的重复 run 逻辑。
- 新增本使用说明与方法映射文档。
- 不删除旧 run 脚本，便于正在运行的任务和历史复现实验继续使用。
- 不改动 MEND 核心实现；MEND 仍由已有脚本负责训练、checkpoint 恢复和评估。
- 不提交实验日志、结果目录、checkpoint、大型数据文件或模型文件。
