"""Serial N=10 gates for BLIP-2 UniKE-ASAM tuning.

This script is a stability/selection gate only. It never asserts a performance
conclusion from N=10.  It runs each candidate serially on GPU 0 and records
complete artifacts so only one justified candidate proceeds to N=100.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE = yaml.safe_load((ROOT / "hparams/UniKE/blip2_ic_unike_asam.yaml").read_text())
RUNS = [
    ("eps005_w025", {"asam_epsilon": 0.05, "asam_weight": 0.25}),
    ("eps010_w025", {"asam_epsilon": 0.10, "asam_weight": 0.25}),
    ("eps005_w050", {"asam_epsilon": 0.05, "asam_weight": 0.50}),
]


def valid_result(path: Path, log: Path, status: Path, n: int) -> bool:
    if not (path.exists() and log.exists() and status.exists()):
        return False
    try:
        result = json.loads(path.read_text())
    except Exception:
        return False
    text = log.read_text(errors="replace")
    return (
        status.read_text().strip() == "EXIT_CODE=0"
        and result.get("sample_count") == n
        and len(result.get("per_case", [])) == n
        and [row.get("case_id") for row in result["per_case"]] == list(range(n))
        and text.count("FINAL_RESULT=") == 1
        and "Traceback" not in text
        and text.count("unike_asam_step=") == n * int(result["hyperparameters"]["n_iter"])
        and "Rephrase sample" not in text
    )


def main():
    config_dir = ROOT / "hparams/UniKE/tuning"
    result_dir = ROOT / "results/UNIKE_BLIP2_IC_ASAM_TUNING_N10"
    log_dir = ROOT / "run_logs/unike_blip2_ic_asam_tuning_n10"
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, override in RUNS:
        config = dict(BASE)
        config.update(override)
        config_path = config_dir / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        output = result_dir / name
        logs = log_dir / name
        output.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        result_path, log_path, status_path = output / "result.json", logs / "run.log", logs / "status.txt"
        if not valid_result(result_path, log_path, status_path, 10):
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": str(ROOT), "UNIKE_TEST_SIZE": "10", "UNIKE_HPARAMS": str(config_path.relative_to(ROOT)),
                "UNIKE_OUTPUT_DIR": str(output.relative_to(ROOT)), "UNIKE_SEED": "42",
            })
            started = time.time()
            with log_path.open("w") as stream:
                process = subprocess.run(["/root/miniconda3/envs/easyedit/bin/python", "run_unike_blip2_ic.py"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            status_path.write_text(f"EXIT_CODE={process.returncode}\nwall_seconds={time.time()-started}\n")
        if not valid_result(result_path, log_path, status_path, 10):
            rows.append({"name": name, "status": "invalid", "config": str(config_path), "result": str(result_path), "log": str(log_path)})
            continue
        result = json.loads(result_path.read_text())
        rows.append({"name": name, "status": "valid_n10_gate", **override, "metrics": result["metrics"], "wall_seconds": result["wall_seconds"], "config": str(config_path), "result": str(result_path), "log": str(log_path)})
    (result_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    print("FINAL_TUNING_GATE=" + json.dumps(rows))


if __name__ == "__main__":
    main()
