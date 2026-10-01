#!/usr/bin/env bash
# LLaVA-OV MEND VQA outer-ASAM, 10000 steps. Keep step10000 for later resume.
set -u
cd /root/PMRL_MLLM
TAG=mend_llavaov_vqa_asam_outer_10k
HP=hparams/TRAINING/MEND/llavaov-7b-vqa-asam-outer-10k.yaml
OUT=/root/PMRL_MLLM/results/MEND_LLAVAOV_VQA_ASAM_OUTER
LOG=run_logs/${TAG}.log
ST=run_logs/${TAG}.status
CKPT_DIR=${OUT}/models/MEND
KEEP=${CKPT_DIR}/llava-onevision.step10000

if pgrep -af '[p]ython .*run_mend_llavaov_vqa_train.py' >/dev/null; then
  printf 'REFUSING_OTHER_TRAINING_PROCESS\n' | tee -a "$LOG"
  exit 97
fi
avail=$(df -B1 --output=avail /root | tail -1)
if [ "$avail" -lt $((12 * 1024 * 1024 * 1024)) ]; then
  printf 'REFUSING_LOW_DISK avail=%s\n' "$(df -h --output=avail /root | tail -1)" | tee -a "$LOG"
  exit 99
fi
mkdir -p "$OUT" run_logs
printf 'START %s config=%s out=%s avail=%s\n' "$(date -Is)" "$HP" "$OUT" "$(df -h --output=avail /root | tail -1)" > "$LOG"

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  PMRL_MEND_HPARAMS="$HP" \
  PMRL_MEND_MAX_ITERS=10000 \
  PMRL_MEND_VAL_INTERVAL=20000 \
  PMRL_MEND_TRAIN_SIZE=1000 \
  PMRL_MEND_VAL_SIZE=100 \
  /root/miniconda3/envs/easyedit/bin/python -u run_mend_llavaov_vqa_train.py >> "$LOG" 2>&1
code=$?
printf 'EXIT_CODE=%s END %s\n' "$code" "$(date -Is)" | tee "$ST" | tee -a "$LOG"

if [ -f "$KEEP" ]; then
  cp -f "$KEEP" "${KEEP}.keep" 2>/dev/null || true
  printf 'KEPT_STEP10000 %s size=%s\n' "$KEEP" "$(stat -c%s "$KEEP" 2>/dev/null || echo unknown)" | tee -a "$LOG"
fi
if [ -f "${CKPT_DIR}/llava-onevision.last" ]; then
  printf 'LAST_CKPT %s\n' "${CKPT_DIR}/llava-onevision.last" | tee -a "$LOG"
fi
exit "$code"
