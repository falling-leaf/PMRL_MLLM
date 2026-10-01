"""Protocol/launcher guards for the single-checkpoint N=100 IC evaluation.

These pin the contract that makes an evaluation row comparable with the frozen
baseline (`results/MEND_LLAVAOV_BASELINE_META_TRAIN/results.json`): same config,
same eval split, archive-loaded validation, `val_steps=100`, shift-corrected
metrics - plus the safety property that the launcher never deletes or rewrites a
checkpoint.  No GPU work; the heavy archive is only read with ``mmap``.
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO / "run_mend_llavaov_ic_eval_one.py"
LAUNCHER = REPO / "run_logs/run_eval_step15000_n100.sh"
VERIFIER = REPO / "tests/verify_ic_checkpoints.py"
CKPT_DIR = REPO / "results/MEND_LLAVAOV_LAP_PMRL_IC_GRAPHFIX_FINAL2/models/MEND"
BASELINE = REPO / "results/MEND_LLAVAOV_BASELINE_META_TRAIN/results.json"


def test_evaluator_protocol_matches_the_baseline():
    src = EVAL_SCRIPT.read_text()
    ast.parse(src)  # compiles
    for line in ("h.eval_only = True", "h.archive = archive", "h.val_steps = eval_size",
                 "h.final_eval = True", "h.save = False", "h.max_iters = 0"):
        assert line in src, f"protocol line missing: {line}"
    # the baseline evaluated this very split, so the rows are pairable
    assert "caption_eval_edit.json" in src
    assert BASELINE.exists()
    base = json.loads(BASELINE.read_text())["results"]
    for key in ("edit/acc_val", "image_rephrase/acc_val", "loc/acc_val", "image_loc/acc_val"):
        assert key in base, f"baseline lacks {key}: not a comparable protocol"
    for key in ("checkpoint_step", "sample_count", "validation_mode", "archive"):
        assert key in src, f"result provenance key missing: {key}"


def test_evaluator_env_contract_is_explicit():
    src = EVAL_SCRIPT.read_text()
    envs = {m.group(1) for m in re.finditer(r'os\.environ\.get\("(\w+)"', src)}
    assert envs == {"PMRL_MEND_EVAL_ARCHIVE", "PMRL_MEND_EVAL_OUT",
                    "PMRL_MEND_EVAL_STEP", "PMRL_MEND_EVAL_SIZE"}
    m_arch = re.search(r"DEFAULT_ARCHIVE = \(([^)]*)\)", src, re.S)
    default_arch = "".join(re.findall(r'"([^"]*)"', m_arch.group(1)))
    assert default_arch.endswith("llava-onevision.step15000")
    assert re.search(r'PMRL_MEND_EVAL_STEP", "15000"', src)


def test_launcher_is_single_shot_and_never_touches_checkpoints():
    assert subprocess.run(["bash", "-n", str(LAUNCHER)]).returncode == 0
    sh = LAUNCHER.read_text()
    assert sh.count("run_mend_llavaov_ic_eval_one.py") == 1
    assert set(re.findall(r"PMRL_MEND_EVAL_\w+(?==)", sh)) == {
        "PMRL_MEND_EVAL_ARCHIVE", "PMRL_MEND_EVAL_OUT", "PMRL_MEND_EVAL_STEP", "PMRL_MEND_EVAL_SIZE"}
    # no destructive or overwriting command anywhere near the archives
    assert not re.search(r"\b(rm|mv|shred|truncate)\b", sh)
    assert ">>" in sh and "run_logs/${TAG}.status" in sh


@pytest.mark.skipif(not CKPT_DIR.exists(), reason="paused run's checkpoints not on this box")
def test_paused_run_archives_are_loadable_and_untouched():
    files = {p.name for p in CKPT_DIR.iterdir() if p.is_file()}
    assert {"llava-onevision", "llava-onevision.bk", "llava-onevision.prevalidation"} <= files
    assert not [f for f in files if ".tmp." in f], "partial write left behind"
    for name in sorted(files):
        p = CKPT_DIR / name
        if name == "llava-onevision.step25000_fresh_optimizer":
            # This deliberately reduced reference copy omits Adam moments and
            # is not one of the paused run's canonical archives.
            continue
        assert 4.1e9 < p.stat().st_size < 4.2e9, f"{name}: unexpected size"
    step15 = CKPT_DIR / "llava-onevision.step15000"
    if step15.exists():  # only present while the hand-made copy is kept
        import torch

        a = torch.load(str(step15), map_location="cpu", weights_only=False, mmap=True)
        assert a["step"] == 15000 and "model" in a and "opt" in a
        del a


def test_checkpoint_verifier_reports_and_fails_correctly(tmp_path):
    import torch

    d = tmp_path / "models"
    d.mkdir()
    torch.save({"step": 7, "model": {"grad_transform.mend.x": torch.ones(2)},
                "opt": {"state": {}}, "lr_opt": {"state": {}}, "val_stats": {"edit/acc_val": 0.5}},
               d / "fake")
    out = tmp_path / "r.json"
    r = subprocess.run([sys.executable, str(VERIFIER), "--dir", str(d), "--json", str(out)],
                       capture_output=True, text=True)
    rec = json.loads(out.read_text())["archives"][0]
    assert r.returncode == 0 and rec["step"] == 7
    assert rec["has_model"] and rec["has_opt"] and rec["has_lr_opt"]
    assert abs(rec["mend_first_param_norm"] - 2 ** 0.5) < 1e-6
    assert subprocess.run([sys.executable, str(VERIFIER), "--dir", str(tmp_path / "nope")],
                          capture_output=True).returncode != 0
