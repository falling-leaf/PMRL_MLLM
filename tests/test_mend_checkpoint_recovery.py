"""Regression guard for preserving a MEND checkpoint before costly validation."""

from pathlib import Path


def test_base_trainer_can_checkpoint_before_validation():
    root = Path(__file__).resolve().parents[1]
    source = (root / "easyeditor/trainer/BaseTrainer.py").read_text(encoding="utf-8")
    assert 'checkpoint_before_validation' in source
    assert 'save_prevalidation_state' in source


def test_qwen2vl_mend_training_enables_prevalidation_checkpoint():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "hparams/TRAINING/MEND/qwen2vl-7b-lap-pmrl.yaml"
    ).read_text(encoding="utf-8")
    assert 'checkpoint_before_validation: true' in source
