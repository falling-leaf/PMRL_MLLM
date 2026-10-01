#!/usr/bin/env python3
"""Run the full 1000-record IC baseline and best enhanced strategy serially."""
import json, os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIZE = 1000
SEED = 42
RUNS = [
    ("baseline", "hparams/WISE/qwen2vl_ic_baseline.yaml"),
    ("best_lar_target_w0p5", "hparams/WISE/qwen2vl_lar_target/lar-target-w0p5.yaml"),
]
RESULT_ROOT = ROOT / "results/QWEN_WISE_IC_FULL"
LOG_ROOT = ROOT / "run_logs/wise_ic_qwen2vl_full"


def run_one(run_id, config):
    out = RESULT_ROOT / run_id
    log_dir = LOG_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": str(ROOT),
        "PMRL_TEST_SIZE": str(SIZE),
        "PMRL_SEED": str(SEED),
        "PMRL_WISE_HPARAMS": config,
        "PMRL_OUTPUT_DIR": str(out.relative_to(ROOT)),
    })
    started = time.time()
    with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            ["/root/miniconda3/envs/easyedit/bin/python", "run_wise_qwen2vl_ic.py"],
            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
        )
    process_wall = time.time() - started
    (log_dir / "status.txt").write_text(
        f"WALL_SECONDS={process_wall}\nEXIT_CODE={proc.returncode}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{run_id} failed: {log_dir / 'run.log'}")
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    if result["sample_count"] != SIZE or len(result["per_case"]) != SIZE:
        raise RuntimeError(f"{run_id}: incomplete result")
    return result


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for run_id, config in RUNS:
        result_path = RESULT_ROOT / run_id / "result.json"
        if result_path.exists():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("sample_count") == SIZE and len(cached.get("per_case", [])) == SIZE:
                summary[run_id] = cached
                continue
        summary[run_id] = run_one(run_id, config)
    output = RESULT_ROOT / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "completed", "sample_count_each": SIZE,
        "baseline": summary["baseline"]["metrics"],
        "enhanced": summary["best_lar_target_w0p5"]["metrics"],
        "summary": str(output),
    }))


if __name__ == "__main__":
    main()
