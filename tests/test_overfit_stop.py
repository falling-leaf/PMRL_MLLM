"""OverfitStopper: cut a MEND VQA ASAM run at the 4k-6k basin, never past 10k."""

from pathlib import Path

from easyeditor.trainer.overfit_stop import (
    OverfitStopper,
    cap_max_iters,
    overfit_stopper_from_config,
    parse_train_log_records,
    replay,
)
from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
    MENDMultimodalTrainingHparams,
)


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "run_logs/mend_llavaov_vqa_asam_outer_10k.log"


def _monotonic_drop(n=12, start=1.0, step=200):
    records = []
    value = start
    for i in range(1, n + 1):
        records.append({"step": i * step, "loss/total_train": value})
        value *= 0.85
    return records


def test_hard_cap_always_stops_at_10000():
    stopper = OverfitStopper()
    for rec in _monotonic_drop(60, start=1.0):
        decision = stopper.update(rec["step"], rec)
        if rec["step"] < 10000:
            assert not decision.stop, rec
        else:
            assert decision.stop
            assert decision.step == 10000
            assert "hard_max_step" in decision.reason
            break
    else:
        raise AssertionError("never reached 10000")


def test_does_not_stop_before_min_step_even_on_rebound():
    stopper = OverfitStopper(min_step=4000, patience_logs=2, rel_rebound=0.1)
    # Deep drop then rebound, all before 4000.
    records = [
        {"step": 200, "loss/total_train": 0.40},
        {"step": 400, "loss/total_train": 0.20},
        {"step": 600, "loss/total_train": 0.10},
        {"step": 800, "loss/total_train": 0.30},
        {"step": 1000, "loss/total_train": 0.35},
        {"step": 1200, "loss/total_train": 0.40},
        {"step": 1400, "loss/total_train": 0.45},
        {"step": 1600, "loss/total_train": 0.50},
    ]
    for rec in records:
        decision = stopper.update(rec["step"], rec)
        assert not decision.stop, rec


def test_stops_on_sustained_rebound_after_min_step():
    stopper = OverfitStopper(
        min_step=4000, patience_logs=3, rel_rebound=0.25, smooth_window=3
    )
    records = [
        {"step": 3600, "loss/total_train": 0.20},
        {"step": 3800, "loss/total_train": 0.12},
        {"step": 4000, "loss/total_train": 0.08},
        {"step": 4200, "loss/total_train": 0.07},
        {"step": 4400, "loss/total_train": 0.065},
        {"step": 4600, "loss/total_train": 0.12},
        {"step": 4800, "loss/total_train": 0.14},
        {"step": 5000, "loss/total_train": 0.16},
        {"step": 5200, "loss/total_train": 0.18},
        {"step": 5400, "loss/total_train": 0.20},
    ]
    decision = replay(records, stopper)
    assert decision is not None and decision.stop
    assert 4000 <= decision.step <= 6000
    assert decision.best_step >= 4000


def test_cap_max_iters_never_exceeds_hard_max():
    assert cap_max_iters(30000) == 10000
    assert cap_max_iters(8000) == 8000
    assert cap_max_iters(None) == 10000


def test_from_config_off_by_default():
    class C:
        overfit_stop = False

    assert overfit_stopper_from_config(C()) is None


def test_yaml_flags_load():
    hp = MENDMultimodalTrainingHparams.from_hparams(
        str(ROOT / "hparams/TRAINING/MEND/llavaov-7b-vqa-asam-outer-10k.yaml")
    )
    assert hp.overfit_stop is True
    assert hp.overfit_stop_min_step == 4000
    assert hp.overfit_stop_hard_max == 10000
    stopper = overfit_stopper_from_config(hp)
    assert stopper is not None
    assert cap_max_iters(30000, stopper.hard_max_step) == 10000


def test_replay_real_10k_log_stops_between_4000_and_6000():
    if not LOG.exists():
        raise AssertionError(f"missing 10k log fixture {LOG}")
    records = parse_train_log_records(LOG.read_text(errors="replace"))
    train = [r for r in records if "loss/total_train" in r]
    assert train and train[-1]["step"] >= 10000
    decision = replay(train, OverfitStopper())
    assert decision is not None and decision.stop
    assert 4000 <= decision.step <= 6000, (
        f"stopped at {decision.step}: {decision.reason}"
    )
    assert decision.step <= 10000


def test_state_dict_roundtrip():
    stopper = OverfitStopper()
    stopper.update(4000, {"loss/total_train": 0.08})
    clone = OverfitStopper()
    clone.load_state_dict(stopper.state_dict())
    assert clone.best_step == 4000
    assert clone.best_value == stopper.best_value
