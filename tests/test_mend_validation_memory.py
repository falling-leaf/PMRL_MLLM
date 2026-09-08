"""Regression test for MEND validation memory discipline."""

from pathlib import Path

from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
    MENDMultimodalTrainingHparams,
)


def test_qwen2vl_mend_disables_full_gradient_diagnostics_by_default():
    root = Path(__file__).resolve().parents[1]
    hparams = MENDMultimodalTrainingHparams.from_hparams(
        str(root / "hparams/TRAINING/MEND/qwen2vl-7b-lap-pmrl.yaml")
    )
    assert hparams.mend_log_grad_diagnostics is False

    source = (root / "easyeditor/trainer/algs/MEND.py").read_text(encoding="utf-8")
    assert 'getattr(self.config, "mend_log_grad_diagnostics", False)' in source
