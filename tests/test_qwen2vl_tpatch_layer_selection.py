from pathlib import Path

from easyeditor.models.transformer_patcher import TransformerPatcherMultimodalHyperParams


def test_qwen2vl_tpatch_uses_validated_editable_layer_23_not_final_layer_27():
    """The Qwen2-VL T-Patcher configs must use the empirically editable layer 23."""
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "qwen2vl_tpatch_baseline.yaml",
        "qwen2vl_tpatch_asam_initial_w025.yaml",
        "qwen2vl_ic_tpatch_baseline.yaml",
        "qwen2vl_ic_tpatch_asam_initial_w025.yaml",
    ):
        hparams = TransformerPatcherMultimodalHyperParams.from_hparams(
            str(root / "hparams" / "Transformer-Patcher" / filename)
        )
        assert hparams.inner_params == [
            "model.language_model.layers.23.mlp.gate_proj.weight",
            "model.language_model.layers.23.mlp.up_proj.weight",
            "model.language_model.layers.23.mlp.down_proj.weight",
        ]
