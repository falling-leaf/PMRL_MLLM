#!/usr/bin/env python3
"""Run at most ten staged Qwen/WISE/IC PMRL tuning experiments."""

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "hparams/WISE/qwen2vl_ic_lap_pmrl.yaml"
CONFIG_DIR = ROOT / "hparams/WISE/qwen2vl_tuning"
RESULT_ROOT = ROOT / "results/QWEN_WISE_IC_PMRL_TUNING_RUNS"
LOG_ROOT = ROOT / "run_logs/wise_ic_qwen2vl_tuning"
SUMMARY_CSV = ROOT / "results/QWEN_WISE_IC_PMRL_TUNING.csv"
BASELINE_RESULT = ROOT / "results/QWEN_WISE_IC_BASELINE_RERUN_N100/result.json"
SEED = 42

# Priority: scale > epsilon > region > number of views > tau2 > tau1.
SCREEN_RUNS = [
    ("t01-scale-0p01", 20, {"pmrl_scale": 0.01}),
    ("t02-scale-0p1", 20, {"pmrl_scale": 0.1}),
    ("t03-eps-0p005-scale-0p01", 20, {"lap_epsilon": 0.005, "pmrl_scale": 0.01}),
    ("t04-eps-0p0005-scale-0p01", 20, {"lap_epsilon": 0.0005, "pmrl_scale": 0.01}),
    ("t05-prompt-region-scale-0p01", 20, {"using_image_embedding": False, "pmrl_scale": 0.01}),
    ("t06-views-10-scale-0p01", 20, {"num_rephrase": 10, "pmrl_scale": 0.01}),
    ("t07-tau2-0p05-scale-0p01", 20, {"pmrl_tau_regularization": 0.05, "pmrl_scale": 0.01}),
    ("t08-tau1-0p01-scale-0p01", 20, {"pmrl_tau_alignment": 0.01, "pmrl_scale": 0.01}),
]

FIELDS = [
    "run_id", "status", "n", "mode", "num_rephrase", "lap_epsilon",
    "pmrl_tau_alignment_tau1", "pmrl_tau_regularization_tau2", "pmrl_scale",
    "using_image_embedding", "acc", "gen_t", "gen_m", "gen_avg", "loc_t",
    "loc_m", "loc_avg", "wall_seconds", "delta_gen_vs_fixed_baseline_n100",
    "seed", "result_path",
]


def load_existing_rows():
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows):
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_row(run_id, status, n, cfg, result=None):
    row = {
        "run_id": run_id,
        "status": status,
        "n": n,
        "mode": "lap_pmrl",
        "num_rephrase": cfg["num_rephrase"],
        "lap_epsilon": cfg["lap_epsilon"],
        "pmrl_tau_alignment_tau1": cfg["pmrl_tau_alignment"],
        "pmrl_tau_regularization_tau2": cfg["pmrl_tau_regularization"],
        "pmrl_scale": cfg["pmrl_scale"],
        "using_image_embedding": str(cfg["using_image_embedding"]).lower(),
        "seed": SEED,
        "result_path": f"results/QWEN_WISE_IC_PMRL_TUNING_RUNS/{run_id}/result.json",
    }
    if result:
        metrics = result["metrics"]
        row.update({
            "acc": metrics["rewrite_acc"], "gen_t": metrics["rephrase_acc"],
            "gen_m": metrics["rephrase_image_acc"], "gen_avg": metrics["gen_avg"],
            "loc_t": metrics["locality_acc"], "loc_m": metrics["multimodal_locality_acc"],
            "loc_avg": metrics["loc_avg"], "wall_seconds": result["wall_seconds"],
        })
    return row


def run_experiment(run_id, n, overrides, rows):
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config.update(overrides)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_dir = RESULT_ROOT / run_id
    log_dir = LOG_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows.append(make_row(run_id, "running", n, config))
    write_rows(rows)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": str(ROOT), "PMRL_TEST_SIZE": str(n), "PMRL_SEED": str(SEED),
        "PMRL_WISE_HPARAMS": str(config_path.relative_to(ROOT)),
        "PMRL_OUTPUT_DIR": str(output_dir.relative_to(ROOT)),
    })
    started = time.time()
    with (log_dir / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            ["/root/miniconda3/envs/easyedit/bin/python", "run_wise_qwen2vl_ic.py"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    wall = time.time() - started
    (log_dir / "status.txt").write_text(
        f"WALL_SECONDS={wall}\nEXIT_CODE={proc.returncode}\n", encoding="utf-8"
    )
    rows.pop()
    if proc.returncode != 0:
        rows.append(make_row(run_id, "failed", n, config))
        write_rows(rows)
        raise RuntimeError(f"{run_id} failed; see {log_dir / 'run.log'}")
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    rows.append(make_row(run_id, "completed", n, config, result))
    write_rows(rows)
    return rows[-1], overrides


def rank_key(row):
    # Main objective Gen Avg; ties favor Gen-M, then locality and lower runtime.
    return (
        float(row["gen_avg"]), float(row["gen_m"]), float(row["loc_avg"]),
        -float(row["wall_seconds"]),
    )


def main():
    baseline = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
    baseline_gen = baseline["metrics"]["gen_avg"]
    rows = [r for r in load_existing_rows() if not r.get("run_id", "").startswith("t")]
    screened = []
    for run_id, n, overrides in SCREEN_RUNS:
        row, override = run_experiment(run_id, n, overrides, rows)
        screened.append((row, override))

    screened.sort(key=lambda item: rank_key(item[0]), reverse=True)
    # Runs 9-10: confirm the two best screened combinations on the full N=100.
    for idx, (screen_row, overrides) in enumerate(screened[:2], start=9):
        run_id = f"t{idx:02d}-confirm-{screen_row['run_id']}"
        run_experiment(run_id, 100, overrides, rows)

    for row in rows:
        if row.get("gen_avg"):
            row["delta_gen_vs_fixed_baseline_n100"] = float(row["gen_avg"]) - baseline_gen
    write_rows(rows)
    print(json.dumps({"status": "completed", "tests": 10, "summary": str(SUMMARY_CSV)}))


if __name__ == "__main__":
    main()
