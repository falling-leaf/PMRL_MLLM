import torch
from torch import nn

from easyeditor.util.pmrl_utils import (
    apply_asam_deltas,
    asam_parameter_deltas,
    request_gen_views,
    select_pmrl_token_views,
)
from easyeditor.models.wise.wise_multimodal_hparams import WISEMultimodalHyperParams
from easyeditor.models.transformer_patcher.transformer_patcher_multimodal_hparams import (
    TransformerPatcherMultimodalHyperParams,
)
from easyeditor.models.mend.mend_multimodal_hparams import MENDMultimodalHparams


def test_asam_floor_moves_zero_initialized_parameters():
    parameter = nn.Parameter(torch.zeros(4))
    gradient = torch.ones_like(parameter)
    collapsed = asam_parameter_deltas([parameter], [gradient], epsilon=0.05, rho=0.0)[0]
    assert collapsed is not None
    assert torch.allclose(collapsed, torch.zeros_like(parameter))
    deltas = asam_parameter_deltas([parameter], [gradient], epsilon=0.05, rho=0.1)
    apply_asam_deltas([parameter], deltas, sign=1.0)
    assert parameter.abs().sum().item() > 0
    apply_asam_deltas([parameter], deltas, sign=-1.0)
    assert torch.allclose(parameter, torch.zeros_like(parameter))


def test_pmrl_token_selection_keeps_visual_tokens_only():
    views = [
        torch.arange(12, dtype=torch.float32).view(1, 4, 3),
        torch.arange(12, 24, dtype=torch.float32).view(1, 4, 3),
    ]
    mask = torch.tensor([[True, True, False, False]])
    selected = select_pmrl_token_views(views, token_mask=mask, pool=False)
    assert selected[0].shape == (2, 3)
    assert torch.equal(selected[0], views[0][0, :2])
    pooled = select_pmrl_token_views(views, token_mask=mask, pool=True)
    assert pooled[0].shape == (1, 3)
    assert torch.allclose(pooled[0], views[0][0, :2].mean(dim=0, keepdim=True))


def test_qwen_wise_vqa_enhancement_uses_visual_tokens_not_answer_joint():
    root = "hparams/WISE/qwen2vl_vqa_best_lar_target.yaml"
    hp = WISEMultimodalHyperParams.from_hparams(root)
    assert hp.using_image_embedding is True
    assert hp.lar_joint_perturbation is False
    assert hp.lar_target_loss_weight == 0.10
    assert hp.pmrl_visual_pooling is False
    assert hp.pmrl_regularization_weight == 0.1


def test_request_gen_views_skip_when_weights_are_zero():
    request = {"prompt": "p", "rephrase_prompt": "r", "image_rephrase": "img", "target": "t"}
    assert request_gen_views(request, 0.0, 0.0) == []
    views = request_gen_views(request, 0.35, 0.5)
    assert len(views) == 2
    assert views[0][0]["prompt"] == "r"
    assert views[0][1] == 0.35
    assert views[1][0]["image"] == "img"
    assert views[1][0]["prompt"] == "p"


def test_llava_tpatch_ic_enhancement_is_visual_prefix_not_joint():
    hp = TransformerPatcherMultimodalHyperParams.from_hparams(
        "hparams/Transformer-Patcher/llavaov_ic_tpatch_asam_w025.yaml"
    )
    assert hp.using_image_embedding is True
    assert hp.lar_joint_perturbation is False
    assert hp.lap_epsilon == 0.005
    assert hp.lar_target_loss_weight == 0.10
    assert hp.pmrl_regularization_weight == 0.1


def test_qwen_and_llava_vqa_tpatch_asam_use_visual_prefix_and_parameter_asam():
    root = "hparams/Transformer-Patcher"
    for filename in (
        "qwen2vl_tpatch_asam_initial_w025.yaml",
        "qwen2vl_ic_tpatch_asam_tune4_iter20.yaml",
        "llavaov_vqa_tpatch_asam_tune6_eps001_w050_iter15.yaml",
    ):
        hp = TransformerPatcherMultimodalHyperParams.from_hparams(f"{root}/{filename}")
        assert hp.using_image_embedding is True
        assert hp.lar_joint_perturbation is False
        assert hp.using_asam is True
        assert hp.asam_replace is True
        assert hp.gen_weight > 0
        assert hp.image_gen_weight > 0
        assert hp.pmrl_regularization_weight == 0.1


def test_qwen_mend_ic_enhancement_aligns_visual_tokens():
    hp = MENDMultimodalHparams.from_hparams("hparams/MEND/qwen2vl_ic_lap_pmrl.yaml")
    assert hp.using_image_embedding is True
    assert hp.pmrl_visual_pooling is False
    assert hp.pmrl_tau_alignment == 0.01
    assert hp.lar_target_loss_weight == 0.10
