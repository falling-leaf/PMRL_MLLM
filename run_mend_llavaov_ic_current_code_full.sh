#!/usr/bin/env bash
# Fresh current-code-only 30k-step LLaVA-OV MEND LAP+PMRL IC run.
set -u
cd /root/PMRL_MLLM
TAG=mend_llavaov_ic_current_code_full_20260914
HP=hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-current-code-full.yaml
OUT=/root/PMRL_MLLM/results/MEND_LLAVAOV_IC_ASAM_CURRENT_CODE_FULL
LOG=run_logs/${TAG}.log
ST=run_logs/${TAG}.status
WATCH=run_logs/${TAG}.diskwatch.log
MANIFEST=${OUT}/run_manifest.json

if [ -e "$ST" ] || [ -e "$MANIFEST" ] || [ -e "$OUT/models" ]; then
  printf 'REFUSING_NONEMPTY_OUTPUT out=%s\n' "$OUT" | tee -a "$LOG"
  exit 98
fi
if pgrep -af 'run_mend_llavaov_ic_train.py|run_mend_llavaov_ic_current_code_full.sh' | grep -v "$$" >/dev/null; then
  printf 'REFUSING_OTHER_TRAINING_PROCESS\n' | tee -a "$LOG"
  exit 97
fi
avail=$(df -B1 --output=avail /root | tail -1)
if [ "$avail" -lt $((15 * 1024 * 1024 * 1024)) ]; then
  printf 'REFUSING_LOW_DISK avail=%s\n' "$(df -h --output=avail /root | tail -1)" | tee -a "$LOG"
  exit 99
fi
mkdir -p "$OUT" run_logs
code_sha=$(sha256sum easyeditor/trainer/BaseTrainer.py easyeditor/trainer/MultimodalTrainer.py easyeditor/trainer/algs/MEND.py easyeditor/trainer/algs/hooks.py easyeditor/trainer/losses.py easyeditor/dataset/coco_caption.py easyeditor/trainer/training_hparams/mend_multimodal_training_hparams.py run_mend_llavaov_ic_train.py "$HP")
python - <<PY
import json, os, subprocess
from pathlib import Path
out=Path("$OUT")
out.joinpath("run_manifest.json").write_text(json.dumps({
 "run":"LLaVA-OV MEND LAP+PMRL IC full training (current code)",
 "started_at":__import__('datetime').datetime.now().astimezone().isoformat(),
 "config":str(Path("$HP").resolve()), "config_archive":None,
 "results_dir":str(out.resolve()), "model": "/root/hugging_cache/llava-onevision-qwen2-7b-ov-hf",
 "train_dataset":"/root/MMEdit/editing-data/caption/caption_train_edit.json",
 "val_dataset":"/root/MMEdit/editing-data/caption/caption_eval_edit.json",
 "train_size":1000,"val_size":1000,"max_iters":30000,
 "using_lap":True,"using_pmrl":True,"num_rephrase":2,"seed":42,
 "code_hashes":dict(line.split(None,1) for line in '''$code_sha'''.splitlines()),
 "environment":{"workers":4,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTORCH_ALLOC_CONF":"expandable_segments:True"},
 "legacy_checkpoint_lineage":None,"disk_avail_at_start":subprocess.check_output(['df','-h','--output=avail','/root'],text=True).splitlines()[-1].strip()
}, indent=2))
PY
printf 'START %s config=%s archive=null out=%s avail=%s\n' "$(date -Is)" "$HP" "$OUT" "$(df -h --output=avail /root | tail -1)" > "$LOG"
(
  while true; do
    printf '%s avail=%s\n' "$(date -Is)" "$(df -h --output=avail /root | tail -1)" >> "$WATCH"
    sleep 600
  done
) & WATCH_PID=$!
trap 'kill "$WATCH_PID" 2>/dev/null || true' EXIT

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  PMRL_MEND_NUM_WORKERS=4 PMRL_MEND_TRAIN_SIZE=1000 PMRL_MEND_VAL_SIZE=1000 \
  PMRL_MEND_HPARAMS="$HP" \
  /root/miniconda3/envs/easyedit/bin/python run_mend_llavaov_ic_train.py >> "$LOG" 2>&1
code=$?
printf 'EXIT_CODE=%s END %s\n' "$code" "$(date -Is)" > "$ST"
printf 'EXIT_CODE=%s\n' "$code" >> "$LOG"
