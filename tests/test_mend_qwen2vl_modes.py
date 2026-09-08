"""Regression tests for Qwen2-VL MEND configuration and mode isolation."""

from pathlib import Path

from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
    MENDMultimodalTrainingHparams,
)


def test_qwen2vl_mend_training_config_targets_real_language_model_parameters():
    root = Path(__file__).resolve().parents[1]
    hparams = MENDMultimodalTrainingHparams.from_hparams(
        str(root / "hparams/TRAINING/MEND/qwen2vl-7b.yaml")
    )

    assert hparams.model_name == "qwen2-vl"
    assert hparams.inner_params
    assert all(name.startswith("model.language_model.layers.") for name in hparams.inner_params)


def test_qwen2vl_mend_training_config_defines_explicit_extra_mode_flags():
    root = Path(__file__).resolve().parents[1]
    hparams = MENDMultimodalTrainingHparams.from_hparams(
        str(root / "hparams/TRAINING/MEND/qwen2vl-7b.yaml")
    )

    assert hparams.using_extra is False
    assert hparams.using_lap is False
    assert hparams.using_pmrl is False


def test_mend_extra_loss_is_not_called_in_baseline_mode():
    """The raw MEND inner loss must remain unchanged unless all gates are on."""
    from types import SimpleNamespace
    from easyeditor.trainer.algs.MEND import MEND

    mend = object.__new__(MEND)
    mend.config = SimpleNamespace(
        model_name="qwen2-vl",
        using_extra=False,
        using_lap=True,
        using_pmrl=True,
    )
    mend._compute_hf_lap_pmrl_loss = lambda *_: (_ for _ in ()).throw(
        AssertionError("baseline must not invoke MEND LAP+PMRL")
    )

    zero = __import__("torch").tensor(0.0)
    result = mend._maybe_mend_extra_loss({}, zero)
    assert result.item() == 0.0
    assert result.device == zero.device


def test_multimodal_executor_normalizes_editor_singleton_request_list():
    from easyeditor.models.mend.mend_main import MendMultimodalRewriteExecutor

    request = {"prompt": "p", "target": "t"}
    assert MendMultimodalRewriteExecutor._singleton_request([request]) == request


def test_multimodal_editor_uses_local_name_for_qwen2vl_loading():
    source = (
        Path(__file__).resolve().parents[1]
        / "easyeditor/editors/multimodal_editor.py"
    ).read_text(encoding="utf-8")
    qwen_branch = source.split('elif "qwen2-vl" in hparams.model_name.lower():', 1)[1]
    qwen_branch = qwen_branch.split("else:", 1)[0]
    assert "Qwen2VLForConditionalGeneration.from_pretrained(" in qwen_branch
    assert qwen_branch.count("hparams.name") >= 2


def test_mend_multimodal_inference_defaults_to_singleton_edits():
    root = Path(__file__).resolve().parents[1]
    from easyeditor.models.mend.mend_multimodal_hparams import MENDMultimodalHparams

    hparams = MENDMultimodalHparams.from_hparams(
        str(root / "hparams/MEND/qwen2vl_ic_baseline.yaml")
    )
    assert hparams.sequential_edit is False
