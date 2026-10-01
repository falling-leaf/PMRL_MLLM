#!/usr/bin/env bash
# LLaVA-OV MEND VQA baseline + OverfitStopper (same cut logic as ASAM early-stop).
set -u
cd /root/PMRL_MLLM
TAG=mend_llavaov_vqa_baseline_earlystop
HP=hparams/TRAINING/MEND/llavaov-7b-vqa-baseline-earlystop.yaml
OUT=/root/PMRL_MLLM/results/MEND_LLAVAOV_VQA_BASELINE_EARLYSTOP
LOG=run_logs/${TAG}.log
ST=run_logs/${TAG}.status
CKPT_DIR=${OUT}/models/MEND
BEST=${CKPT_DIR}/llava-onevision
EVAL_OUT=/root/PMRL_MLLM/results/MEND_LLAVAOV_VQA_BASELINE_EARLYSTOP_N100
EVAL_LOG=run_logs/${TAG}_eval.log
PY=/root/miniconda3/envs/easyedit/bin/python

if pgrep -af '[p]ython .*run_mend_llavaov_vqa_train.py' >/dev/null; then
  printf 'REFUSING_OTHER_TRAINING_PROCESS\n' | tee -a "$LOG"
  exit 97
fi
avail=$(df -B1 --output=avail /root | tail -1)
if [ "$avail" -lt $((16 * 1024 * 1024 * 1024)) ]; then
  printf 'REFUSING_LOW_DISK avail=%s\n' "$(df -h --output=avail /root | tail -1)" | tee -a "$LOG"
  exit 99
fi
mkdir -p "$OUT" run_logs
{
  printf 'START %s config=%s out=%s avail=%s gpu=' "$(date -Is)" "$HP" "$OUT" "$(df -h --output=avail /root | tail -1)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$LOG"

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  PMRL_MEND_HPARAMS="$HP" \
  PMRL_MEND_MAX_ITERS=10000 \
  PMRL_MEND_VAL_INTERVAL=20000 \
  PMRL_MEND_TRAIN_SIZE=1000 \
  PMRL_MEND_VAL_SIZE=100 \
  "$PY" -u run_mend_llavaov_vqa_train.py >> "$LOG" 2>&1
code=$?
printf 'EXIT_CODE=%s END %s\n' "$code" "$(date -Is)" | tee "$ST" | tee -a "$LOG"

if [ -f "$BEST" ]; then
  cp -f "$BEST" "${BEST}.keep" 2>/dev/null || true
  printf 'KEPT %s size=%s\n' "$BEST" "$(stat -c%s "$BEST" 2>/dev/null || echo unknown)" | tee -a "$LOG"
fi
if [ -f "${CKPT_DIR}/llava-onevision.last" ]; then
  cp -f "${CKPT_DIR}/llava-onevision.last" "${CKPT_DIR}/llava-onevision.last.keep" 2>/dev/null || true
  printf 'KEPT %s size=%s\n' "${CKPT_DIR}/llava-onevision.last" "$(stat -c%s "${CKPT_DIR}/llava-onevision.last" 2>/dev/null || echo unknown)" | tee -a "$LOG"
fi

if [ -f "$BEST" ]; then
  printf 'START %s archive=best\n' "$(date -Is)" > "$EVAL_LOG"
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PY" -u - >> "$EVAL_LOG" 2>&1 <<PY
from run_mend_llavaov_vqa_eval import run_eval
import json
from pathlib import Path
archive = "${BEST}"
out = "${EVAL_OUT}"
info = run_eval(
    config_path="${HP}",
    archive_path=archive,
    results_dir=out,
    eval_size=100,
)
path = Path(out) / "result.json"
payload = {
    "archive": archive,
    "config": "${HP}",
    "sample_count": 100,
    "results": {k: float(v) if hasattr(v, "real") or isinstance(v, float) else v for k, v in info.items()},
}
path.write_text(json.dumps(payload, indent=2))
print("Results saved to", path, flush=True)
PY
  eval_code=$?
  printf 'EVAL_EXIT=%s END %s\n' "$eval_code" "$(date -Is)" | tee -a "$EVAL_LOG" | tee -a "$LOG"
fi
exit "$code"
