"""Behavioral tests for Transformer-Patcher's paired FFN neuron expansion."""

from collections import UserDict
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from easyeditor.models.transformer_patcher.layers import (
    ExpandedLinearInput,
    ExpandedLinearOutput,
)
from easyeditor.models.transformer_patcher.transformer_patcher_main import (
    TransformerPatcherMultimodal,
)
from easyeditor.models.transformer_patcher.transformer_patcher_multimodal_hparams import (
    TransformerPatcherMultimodalHyperParams,
)


def test_paired_expanded_ffn_preserves_base_and_routes_new_neuron():
    torch.manual_seed(11)
    fc1 = torch.nn.Linear(3, 4, bias=True)
    fc2 = torch.nn.Linear(4, 2, bias=True)
    patched_fc1 = ExpandedLinearOutput(fc1, add_neuron_num=1)
    patched_fc2 = ExpandedLinearInput(fc2, add_neuron_num=1)

    x = torch.randn(2, 5, 3)
    base = fc2(F.relu(fc1(x)))
    assert torch.allclose(patched_fc2(F.relu(patched_fc1(x))), base)

    with torch.no_grad():
        patched_fc1.extra_output.weight.fill_(0.25)
        patched_fc1.extra_output.bias.fill_(0.5)
        patched_fc2.extra_input.weight.fill_(0.4)

    expected = base + F.linear(
        F.relu(F.linear(x, patched_fc1.extra_output.weight, patched_fc1.extra_output.bias)),
        patched_fc2.extra_input.weight,
        None,
    )
    actual = patched_fc2(F.relu(patched_fc1(x)))
    assert torch.allclose(actual, expected)
    assert all(not p.requires_grad for p in patched_fc1.linear.parameters())
    assert all(not p.requires_grad for p in patched_fc2.linear.parameters())
    assert patched_fc1.extra_output.weight.requires_grad
    assert patched_fc2.extra_input.weight.requires_grad


def test_transformer_patcher_requires_paired_opt_fc1_fc2():
    assert TransformerPatcherMultimodal._resolve_pair(
        [
            "opt_model.model.decoder.layers.23.fc1.weight",
            "opt_model.model.decoder.layers.23.fc2.weight",
        ]
    ) == [
        "opt_model.model.decoder.layers.23.fc1",
        "opt_model.model.decoder.layers.23.fc2",
    ]
    try:
        TransformerPatcherMultimodal._resolve_pair(["only.one.weight"])
    except ValueError as error:
        assert "exactly" in str(error)
    else:
        raise AssertionError("unpaired FFN parameters must be rejected")


def test_transformer_patcher_recognizes_llama_and_qwen_swiglu_mlp_triplets():
    llama = [
        "llama_model.model.layers.31.mlp.gate_proj.weight",
        "llama_model.model.layers.31.mlp.up_proj.weight",
        "llama_model.model.layers.31.mlp.down_proj.weight",
    ]
    qwen = [
        "model.language_model.layers.23.mlp.gate_proj.weight",
        "model.language_model.layers.23.mlp.up_proj.weight",
        "model.language_model.layers.23.mlp.down_proj.weight",
    ]
    assert TransformerPatcherMultimodal._resolve_swiglu_triplet(llama) == [
        "llama_model.model.layers.31.mlp.gate_proj",
        "llama_model.model.layers.31.mlp.up_proj",
        "llama_model.model.layers.31.mlp.down_proj",
    ]
    assert TransformerPatcherMultimodal._resolve_swiglu_triplet(qwen) == [
        "model.language_model.layers.23.mlp.gate_proj",
        "model.language_model.layers.23.mlp.up_proj",
        "model.language_model.layers.23.mlp.down_proj",
    ]


def test_swiglu_expanded_triplet_preserves_base_before_training():
    torch.manual_seed(7)
    gate = torch.nn.Linear(3, 5, bias=False)
    up = torch.nn.Linear(3, 5, bias=False)
    down = torch.nn.Linear(5, 2, bias=False)
    expanded_gate = ExpandedLinearOutput(gate, add_neuron_num=1)
    expanded_up = ExpandedLinearOutput(up, add_neuron_num=1)
    expanded_down = ExpandedLinearInput(down, add_neuron_num=1)
    x = torch.randn(2, 4, 3)
    base = down(F.silu(gate(x)) * up(x))
    patched = expanded_down(F.silu(expanded_gate(x)) * expanded_up(x))
    assert torch.allclose(patched, base)


def test_expanded_layers_preserve_half_precision_of_frozen_linear():
    linear = torch.nn.Linear(3, 4, bias=False).half()
    expanded_output = ExpandedLinearOutput(linear, add_neuron_num=1)
    expanded_input = ExpandedLinearInput(linear, add_neuron_num=1)
    assert expanded_output.extra_output.weight.dtype is torch.float16
    assert expanded_input.extra_input.weight.dtype is torch.float16


def test_tpatch_hparams_define_explicit_baseline_and_pmrl_modes():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    baseline = TransformerPatcherMultimodalHyperParams.from_hparams(
        str(root / "hparams/Transformer-Patcher/blip2_ic_tpatch.yaml")
    )
    enhanced = TransformerPatcherMultimodalHyperParams.from_hparams(
        str(root / "hparams/Transformer-Patcher/blip2_ic_tpatch_pmrl.yaml")
    )
    assert (baseline.using_extra, baseline.using_lap, baseline.using_pmrl) == (
        False,
        False,
        False,
    )
    assert (enhanced.using_extra, enhanced.using_lap, enhanced.using_pmrl) == (
        True,
        True,
        True,
    )
    assert enhanced.num_rephrase > 1


def test_tpatch_baseline_does_not_call_pmrl_helper():
    patcher = object.__new__(TransformerPatcherMultimodal)
    torch.nn.Module.__init__(patcher)
    patcher.config = SimpleNamespace(
        using_extra=False, using_lap=False, using_pmrl=False, locality_weight=1.0
    )
    patcher._compute_lap_pmrl_loss = lambda *_: (_ for _ in ()).throw(
        AssertionError("PMRL path must not run in baseline mode")
    )
    patcher._base_losses = lambda *_: (
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.ones(1, 2, 4),
        torch.ones(1, 2, dtype=torch.long),
        torch.full((1, 2), -100, dtype=torch.long),
    )
    total = patcher._loss_with_optional_pmrl({}, {})
    assert total.item() == 5.0


def test_randomized_projected_lap_views_are_diverse_and_within_budget():
    patcher = object.__new__(TransformerPatcherMultimodal)
    torch.nn.Module.__init__(patcher)
    torch.manual_seed(17)
    base = torch.zeros(1, 6, 4)
    gradient = torch.ones_like(base)
    views = patcher._build_lap_perturbations(
        base,
        gradient,
        perturb_end=3,
        num_views=5,
        epsilon=0.03,
        random_start=True,
        pgd_steps=1,
        step_size=0.03,
    )
    assert len(views) == 5
    for view in views:
        delta = view - base
        assert torch.all(delta[:, 3:, :] == 0)
        assert delta.float().flatten(1).norm(dim=1).max().item() <= 0.03001
    assert any(not torch.equal(views[0], view) for view in views[1:])


def test_randomized_lap_keeps_zero_float16_gradients_finite():
    patcher = object.__new__(TransformerPatcherMultimodal)
    torch.nn.Module.__init__(patcher)
    torch.manual_seed(19)
    base = torch.zeros(1, 51, 4096, dtype=torch.float16)
    gradient = torch.zeros_like(base)
    views = patcher._build_lap_perturbations(
        base, gradient, perturb_end=32, num_views=5, epsilon=0.005,
        random_start=True, pgd_steps=1, step_size=0.0005,
    )
    assert all(torch.isfinite(view).all() for view in views)


def test_tpatch_hparams_expose_variant_target_consistency_weight():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    enhanced = TransformerPatcherMultimodalHyperParams.from_hparams(
        str(root / "hparams/Transformer-Patcher/random_lap/tpatch_pmrl_random_eps0015_step00015.yaml")
    )
    assert enhanced.lar_target_loss_weight == 0.0


def test_tpatch_vqa_configs_define_isolated_baseline_and_random_projected_pmrl():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    baseline = TransformerPatcherMultimodalHyperParams.from_hparams(
        str(root / "hparams/Transformer-Patcher/blip2_vqa_tpatch.yaml")
    )
    enhanced = TransformerPatcherMultimodalHyperParams.from_hparams(
        str(root / "hparams/Transformer-Patcher/blip2_vqa_tpatch_pmrl_random_eps0015_step00015.yaml")
    )
    target_consistency = TransformerPatcherMultimodalHyperParams.from_hparams(
        str(root / "hparams/Transformer-Patcher/blip2_vqa_tpatch_pmrl_random_eps0015_step00015_w025.yaml")
    )
    assert (baseline.using_extra, baseline.using_lap, baseline.using_pmrl) == (False, False, False)
    assert (enhanced.using_extra, enhanced.using_lap, enhanced.using_pmrl) == (True, True, True)
    assert enhanced.lar_random_start is True
    assert enhanced.lap_epsilon == 0.015
    assert enhanced.lar_step_size == 0.0015
    assert enhanced.export_lap_samples is False
    assert target_consistency.lar_target_loss_weight == 0.25
    assert target_consistency.lap_epsilon == enhanced.lap_epsilon
    assert target_consistency.lar_step_size == enhanced.lar_step_size


def test_tpatch_minigpt4_config_uses_swiglu_triplet_and_explicit_modes():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    baseline = TransformerPatcherMultimodalHyperParams.from_hparams(
        str(root / "hparams/Transformer-Patcher/minigpt4_ic_tpatch.yaml")
    )
    enhanced = TransformerPatcherMultimodalHyperParams.from_hparams(
        str(root / "hparams/Transformer-Patcher/minigpt4_ic_tpatch_asam_w025.yaml")
    )
    assert len(baseline.inner_params) == 3
    assert baseline.inner_params[-1].endswith(".down_proj.weight")
    assert (baseline.using_extra, baseline.using_lap, baseline.using_pmrl) == (False, False, False)
    assert (enhanced.using_extra, enhanced.using_lap, enhanced.using_pmrl) == (True, True, True)
    assert enhanced.lar_target_loss_weight == 0.25


def test_tpatch_hf_target_loss_uses_explicit_batch_labels():
    logits = torch.tensor([[[0.0, 3.0], [3.0, 0.0], [0.0, 3.0]]])
    labels = torch.tensor([[-100, 1, 0]])
    loss = TransformerPatcherMultimodal._hf_target_loss(logits, labels)
    assert loss.item() < 0.1


def test_tpatch_hf_forward_unpacks_mapping_batchfeature_equivalent():
    patcher = object.__new__(TransformerPatcherMultimodal)
    torch.nn.Module.__init__(patcher)
    patcher.config = SimpleNamespace(model_name="/root/hugging_cache/llava-onevision-qwen2-7b-ov-hf")
    observed = {}

    def model(**kwargs):
        observed.update(kwargs)
        return "ok"

    patcher.model = model
    assert patcher._forward_inputs(UserDict(input_ids=torch.tensor([[1]]))) == "ok"
    assert torch.equal(observed["input_ids"], torch.tensor([[1]]))


def test_tpatch_hf_model_path_is_not_treated_as_a_legacy_wrapper():
    patcher = object.__new__(TransformerPatcherMultimodal)
    torch.nn.Module.__init__(patcher)
    patcher.config = SimpleNamespace(model_name="/root/hugging_cache/llava-onevision-qwen2-7b-ov-hf")
    assert patcher._is_legacy_wrapper() is False
    patcher.config = SimpleNamespace(model_name="minigpt4")
    assert patcher._is_legacy_wrapper() is True


def test_tpatch_hf_config_parses_joint_perturbation_and_target_only_labels():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config = TransformerPatcherMultimodalHyperParams.from_hparams(
        str(root / "hparams/Transformer-Patcher/llavaov_ic_tpatch_asam_w025.yaml")
    )
    assert config.lar_joint_perturbation is True
    assert config.objective_optimization == "only_label"


def test_tpatch_hf_lap_perturbations_only_change_selected_tokens_within_budget():
    patcher = object.__new__(TransformerPatcherMultimodal)
    torch.nn.Module.__init__(patcher)
    torch.manual_seed(19)
    base = torch.zeros(1, 5, 3)
    grad = torch.ones_like(base)
    mask = torch.tensor([[True, False, True, False, False]])
    views = patcher._build_masked_lap_perturbations(
        base, grad, mask, num_views=3, epsilon=0.02, random_start=True, pgd_steps=1, step_size=0.002
    )
    assert len(views) == 3
    for delta in views:
        assert torch.all(delta[:, ~mask[0], :] == 0)
        assert delta.float().flatten(1).norm(dim=1).max().item() <= 0.02001
