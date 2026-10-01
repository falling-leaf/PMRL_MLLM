"""Build the machine-readable timing record for the LLaVA-OV MEND IC (ASAM) runs.

Reads the MEND_PROFILE logs produced on 2026-09-12 (see
docs/mend_llavaov_ic_timing_gpu_measurement.md) and emits a JSON record with:

* per-run steady-state step time (from the trainer's step echoes) and the
  de-duplicated section breakdown,
* the data-pipeline / overhead residual (wall clock - section sum),
* the validation-event cost (pre-validation checkpoint write + 100 val samples
  + best-model write) measured in run ``perf_perf_n24_val``,
* the 30 000-step ETA built from those measurements.

Usage:
    python tests/mend_ic_timing_report.py [--json OUT] [--md OUT]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "run_logs"

TS = re.compile(r"^(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - INFO - [^-]* - ?(.*)$")
STEP = re.compile(r"^Step (\d+):$")
HDR = re.compile(r"^MEND_PROFILE step=(\d+) steps_covered=(\d+) measured_total_s=([\d.]+)")
SEC = re.compile(r"^MEND_PROFILE\s+(\S+)\s+sum=\s*([\d.]+)s n=(\d+)\s+mean=\s*([\d.]+)ms")

# sections that are nested inside another section's timer (double counted)
PARENT = {"edit.lap_probe": "edit.loss", "edit.lap_view": "edit.loss", "edit.pmrl": "edit.loss"}

# frozen experiment reference (pre-optimisation run of the same config)
HISTORIC = {
    "log": "mend_llavaov_ic_graphfix_final2.log",
    "first_step": 100,
    "first_ts": "09/10/2026 17:14:19",
    "last_step": 2200,
    "last_ts": "09/10/2026 21:34:10",
}


def stamp(line):
    m = TS.match(line)
    return datetime.strptime(m.group(1), "%m/%d/%Y %H:%M:%S") if m else None


def body_of(line):
    """Strip the ``<ts> - INFO - <logger> - `` prefix when present."""
    m = TS.match(line)
    return m.group(2).strip() if m else line.strip()


def parse_log(path):
    echoes, windows = [], []
    cur = None
    for line in Path(path).read_text(errors="replace").split("\n"):
        ts = stamp(line)
        body = body_of(line)
        h = HDR.match(body)
        if h:
            cur = {"step": int(h.group(1)), "covered": int(h.group(2)),
                   "raw_total_s": float(h.group(3)), "sections": {}, "ts": ts}
            windows.append(cur)
            continue
        s = SEC.match(body)
        if s and cur is not None:
            cur["sections"][s.group(1)] = {"sum_s": float(s.group(2)), "n": int(s.group(3))}
            continue
        st = STEP.match(body)
        if st and ts:
            echoes.append({"step": int(st.group(1)), "ts": ts})
    return {"echoes": echoes, "windows": windows}


def steady_step_time(echoes):
    """Steady-state s/step from consecutive step echoes (skips the warm-up)."""
    deltas = []
    for prev, cur in zip(echoes, echoes[1:]):
        n = cur["step"] - prev["step"]
        if n <= 0:
            continue
        deltas.append((cur["ts"] - prev["ts"]).total_seconds() / n)
    if not deltas:
        return {}
    # drop the first interval: it contains CUDA/allocator warm-up
    steady = deltas[1:] if len(deltas) > 1 else deltas
    return {
        "intervals": len(deltas),
        "steady_intervals": len(steady),
        "s_per_step_mean": round(sum(steady) / len(steady), 3),
        "s_per_step_min": round(min(steady), 3),
        "s_per_step_max": round(max(steady), 3),
    }


def training_windows(windows):
    """Training windows carry the LAP probe; validation windows do not."""
    return [w for w in windows if "edit.lap_probe" in w["sections"]]


def section_breakdown(windows):
    windows = training_windows(windows)
    if not windows:
        return {}
    last = windows[-1]
    cov = last["covered"]
    return {
        k: round(v["sum_s"] / cov * 1000, 1)
        for k, v in sorted(last["sections"].items(), key=lambda kv: -kv[1]["sum_s"])
    }


def dedup_per_step(windows, skip_warmup=True):
    windows = training_windows(windows)
    ws = windows[1:] if (skip_warmup and len(windows) > 1) else windows
    if not ws:
        return None
    return round(sum(
        sum(v["sum_s"] for k, v in w["sections"].items() if k not in PARENT) / w["covered"]
        for w in ws) / len(ws), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "results" / "mend_llavaov_ic_perf_timing.json"))
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    runs = {
        "A_frozen_w0": ("mend_llavaov_ic_perf_n100_w0.log", "frozen config, dataloader_num_workers=0, profiled", 100),
        "C_frozen_w4": ("perf_perf_n40_w4.log", "frozen config, dataloader_num_workers=4, profiled", 40),
        "D_legacy_norm": ("perf_perf_n20_seqnorm.log", "MEND_NORM_SEQUENTIAL=1 (pre-optimisation Welford loop)", 20),
        "F_validation": ("perf_perf_n24_val.log", "frozen config + one validation at step 20", 24),
        "G_eq_block": ("perf_perf_eq_block.log", "log_interval=1, block Welford (3 steps)", 3),
        "H_eq_seq": ("perf_perf_eq_seq.log", "log_interval=1, legacy Welford (3 steps)", 3),
        "RA_repeat": ("perf_perf_rep_a.log", "repeatability pair, identical settings a", 20),
        "RB_repeat": ("perf_perf_rep_b.log", "repeatability pair, identical settings b", 20),
    }
    rec = {"generated": datetime.now().isoformat(timespec="seconds"),
           "box": {"gpu": "NVIDIA RTX 6000D (85 GB)", "torch": "2.9.1+cu128",
                   "transformers": "4.57.1", "python": "3.10 (easyedit env)"},
           "config": "hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-graphfix-final2.yaml",
           "config_sha_of_timing_copy": "llavaov-7b-lap-pmrl-ic-graphfix-final2-timing.yaml",
           "runs": {}}

    for tag, (fname, note, nsteps) in runs.items():
        path = LOGS / fname
        if not path.exists():
            rec["runs"][tag] = {"log": fname, "note": note, "status": "missing"}
            continue
        p = parse_log(path)
        st = steady_step_time(p["echoes"])
        rec["runs"][tag] = {
            "log": fname, "note": note, "steps": nsteps,
            "exit": (LOGS / fname.replace(".log", ".status")).read_text().strip()
            if (LOGS / fname.replace(".log", ".status")).exists() else None,
            "step_time": st,
            "section_sum_s_per_step": dedup_per_step(p["windows"]),
            "sections_ms_per_step": section_breakdown(p["windows"]),
        }
        if st:
            rec["runs"][tag]["overhead_s_per_step"] = round(
                st["s_per_step_mean"] - (rec["runs"][tag]["section_sum_s_per_step"] or 0), 3)

    # validation event cost (run F): prevalidation write -> 100 val samples -> save
    fpath = LOGS / "perf_perf_n24_val.log"
    if fpath.exists():
        txt = fpath.read_text(errors="replace")
        events = {}
        for key, pat in [("prevalidation_write_logged", "Saving pre-validation model"),
                         ("best_model_write_done", "Write complete."),
                         ("saving_best_model", "Saving model to"),
                         ("training_complete", "Training complete after")]:
            m = [l for l in txt.split("\n") if pat in l and stamp(l)]
            events[key] = stamp(m[-1]).isoformat() if m else None
        # every profile dump during validation covers MEND_PROFILE_EVERY=10
        # val samples, so the wall clock per val sample is measurable directly
        val_windows = [w for w in parse_log(fpath)["windows"]
                       if w["sections"] and "edit.lap_probe" not in w["sections"]]
        per_sample = None
        if len(val_windows) > 2:
            first, last = val_windows[0], val_windows[-1]
            span = (last["ts"] - first["ts"]).total_seconds() + first["raw_total_s"]
            per_sample = round(span / (10 * len(val_windows)), 3)
        events["val_samples"] = 100
        events["val_s_per_sample"] = per_sample
        events["val_loop_s"] = round(100 * per_sample, 1) if per_sample else None
        rec["validation_event"] = events

    # derived ETA for the configured 30 000-step run
    frozen = rec["runs"]["A_frozen_w0"]["step_time"]["s_per_step_mean"]
    w4 = rec["runs"]["C_frozen_w4"]["step_time"]["s_per_step_mean"]
    val = rec["validation_event"]
    val_block = (val.get("val_loop_s") or 0) + 62  # + 2 checkpoint writes (~31 s each)
    def eta(s_per_step, final_eval_samples):
        train = 30000 * s_per_step
        val5 = 5 * val_block
        final = 15 + final_eval_samples * (val.get("val_s_per_sample") or 0)
        total = train + val5 + final + 67
        return {"s_per_step": s_per_step, "train_h": round(train / 3600, 2),
                "mid_validations_5x_h": round(val5 / 3600, 2),
                "final_eval_h": round(final / 3600, 2),
                "total_h": round(total / 3600, 2)}
    rec["eta_30000_steps"] = {
        "frozen_workers0": eta(frozen, 1000),
        "opt_in_workers4": eta(w4, 1000),
        "frozen_workers0_final_eval_100": eta(frozen, 100),
        "historic_pre_optimisation_s_per_step": round(
            (datetime.strptime(HISTORIC["last_ts"], "%m/%d/%Y %H:%M:%S")
             - datetime.strptime(HISTORIC["first_ts"], "%m/%d/%Y %H:%M:%S")).total_seconds()
            / (HISTORIC["last_step"] - HISTORIC["first_step"]), 3),
    }
    # same accounting for the pre-optimisation speed (validation cost scales
    # with the step cost: measured 2.954 s/val sample at 5.487 s/step)
    old = rec["eta_30000_steps"]["historic_pre_optimisation_s_per_step"]
    old_val_sample = round((val.get("val_s_per_sample") or 0) * old / frozen, 3)
    old_block = old_val_sample * 100 + 62
    rec["eta_30000_steps"]["historic_pre_optimisation"] = {
        "s_per_step": old,
        "train_h": round(30000 * old / 3600, 2),
        "mid_validations_5x_h": round(5 * old_block / 3600, 2),
        "final_eval_h": round((15 + 1000 * old_val_sample) / 3600, 2),
        "total_h": round((30000 * old + 5 * old_block + 15 + 1000 * old_val_sample + 67) / 3600, 2),
        "val_s_per_sample": old_val_sample,
    }
    rec["eta_30000_steps"]["saving_h"] = round(
        rec["eta_30000_steps"]["historic_pre_optimisation"]["total_h"]
        - rec["eta_30000_steps"]["frozen_workers0"]["total_h"], 2)
    rec["eta_30000_steps"]["saving_pct"] = round(
        100 * rec["eta_30000_steps"]["saving_h"]
        / rec["eta_30000_steps"]["historic_pre_optimisation"]["total_h"], 1)
    rec["disk"] = {"note": "each MEND checkpoint is 3.87 GiB; a validation event "
                           "leaves <name>, <name>.bk and <name>.prevalidation "
                           "(peak 3 x 3.87 GiB = 11.6 GiB per run dir)"}
    rec["historic_pre_optimisation"] = HISTORIC
    rec["repeatability"] = {
        "note": "two identical invocations (rep_a/rep_b) of the optimised path; "
                "deviation between them is the pipeline noise floor that any "
                "code-variant comparison must be judged against",
        "logs": ["perf_perf_rep_a.log", "perf_perf_rep_b.log"],
    }

    md = []
    if args.md:
        md = ["# placeholder"]

    # step-level metrics for the equivalence / repeatability discussion
    def echo_metrics(fname):
        out = {}
        for e in parse_log(LOGS / fname)["echoes"]:
            out[e["step"]] = e
        lines = (LOGS / fname).read_text(errors="replace").split("\n")
        vals = {}
        for i, l in enumerate(lines):
            m = STEP.match(body_of(l)) if stamp(l) else None
            if m:
                vals[int(m.group(1))] = {
                    k: float(v) for k, v in re.findall(r"([a-zA-Z_/]+):\s*(-?[\d.]+)", lines[i + 1])
                }
        return vals

    eq = {}
    for tag, fname in [("A_optimised_100st", "mend_llavaov_ic_perf_n100_w0.log"),
                       ("D_legacy_norm_20st", "perf_perf_n20_seqnorm.log"),
                       ("G_block_3st_log1", "perf_perf_eq_block.log"),
                       ("H_legacy_3st_log1", "perf_perf_eq_seq.log"),
                       ("RA_repeat_a", "perf_perf_rep_a.log"),
                       ("RB_repeat_b", "perf_perf_rep_b.log")]:
        eq[tag] = echo_metrics(fname)
    rec["step_metrics"] = eq
    rel = {}
    for pair, (x, y) in {"repeat_a_vs_repeat_b": ("RA_repeat_a", "RB_repeat_b"),
                         "block_vs_legacy_20st": ("A_optimised_100st", "D_legacy_norm_20st"),
                         "block_vs_legacy_step1to3": ("G_block_3st_log1", "H_legacy_3st_log1")}.items():
        rel[pair] = {}
        for step in sorted(set(eq[x]) & set(eq[y])):
            for k in ("loss/edit_train", "loss/total_edit_train", "edit/acc_train",
                      "loc/acc_train", "grad_train", "diag/outer_nonzero_train"):
                a, b = eq[x][step].get(k), eq[y][step].get(k)
                if a is None or b is None:
                    continue
                rel[pair][f"{k}@{step}"] = round(abs(a - b) / max(abs(a), 1e-30), 10)
    rec["relative_deviations"] = rel

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    sys.exit(main())
