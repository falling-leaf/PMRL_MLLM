from pathlib import Path

from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
    MENDMultimodalTrainingHparams,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "hparams/TRAINING/MEND/llavaov-7b-vqa.yaml"


def test_llava_vqa_mend_uses_normalized_gradients():
    hp = MENDMultimodalTrainingHparams.from_hparams(str(CONFIG))
    assert hp.model_name == "llava-onevision"
    assert hp.norm is True
    assert hp.using_extra is False
    assert hp.using_lap is False
    assert hp.using_pmrl is False
    assert hp.inner_params == [
        "model.language_model.layers.27.mlp.up_proj.weight",
        "model.language_model.layers.27.mlp.down_proj.weight",
    ]
