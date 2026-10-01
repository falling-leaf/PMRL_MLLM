#!/usr/bin/env bash
set -o pipefail
cd /root/PMRL_MLLM
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/root/PMRL_MLLM
export PMRL_TEST_SIZE=100
run_one() {
  local mode="$1" config="$2" output="$3"
  local logdir="run_logs/wise_ic_qwen2vl_n100/${mode}"
  mkdir -p "$logdir" "$output"
  export PMRL_WISE_HPARAMS="$config" PMRL_OUTPUT_DIR="$output"
  local start end rc
  start=$(date +%s)
  /root/miniconda3/envs/easyedit/bin/python run_wise_qwen2vl_ic.py 2>&1 | tee "$logdir/run.log"
  rc=${PIPESTATUS[0]}
  end=$(date +%s)
  printf 'WALL_SECONDS=%s\nEXIT_CODE=%s\n' "$((end-start))" "$rc" > "$logdir/status.txt"
  return "$rc"
}
run_one baseline hparams/WISE/qwen2vl_ic_baseline.yaml results/WISE_IC_QWEN2VL_BASELINE_N100 || exit $?
run_one lap_pmrl hparams/WISE/qwen2vl_ic_lap_pmrl.yaml results/WISE_IC_QWEN2VL_LAP_PMRL_N100
