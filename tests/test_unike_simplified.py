import torch
from torch import nn

from easyeditor.models.unike_simplified import ExpandedSwiGLUMLP


def test_expanded_swiglu_preserves_function_to_small_tolerance_at_initialization():
    torch.manual_seed(0)
    original = nn.Module()
    original.gate_proj = nn.Linear(4, 6, bias=False)
    original.up_proj = nn.Linear(4, 6, bias=False)
    original.down_proj = nn.Linear(6, 4, bias=False)
    expanded = ExpandedSwiGLUMLP(original, add_neuron_num=3)
    x = torch.randn(2, 5, 4)
    base = original.down_proj(torch.nn.functional.silu(original.gate_proj(x)) * original.up_proj(x))
    # extra_down starts deliberately small-but-nonzero so gradients reach
    # gate/up on step 1; the base function must nevertheless be preserved closely.
    assert (expanded(x) - base).abs().max() < 1e-2


def test_expanded_swiglu_extra_parameters_are_trainable():
    original = nn.Module()
    original.gate_proj = nn.Linear(4, 6, bias=False)
    original.up_proj = nn.Linear(4, 6, bias=False)
    original.down_proj = nn.Linear(6, 4, bias=False)
    expanded = ExpandedSwiGLUMLP(original, add_neuron_num=3)
    assert all(p.requires_grad for n, p in expanded.named_parameters() if n.startswith("extra_"))


def test_expanded_swiglu_extra_down_is_nonzero_for_step_one_gradient_flow():
    original = nn.Module()
    original.gate_proj = nn.Linear(4, 6, bias=False)
    original.up_proj = nn.Linear(4, 6, bias=False)
    original.down_proj = nn.Linear(6, 4, bias=False)
    expanded = ExpandedSwiGLUMLP(original, add_neuron_num=3)
    assert torch.count_nonzero(expanded.extra_down.weight) > 0
    x = torch.randn(2, 5, 4)
    expanded(x).square().mean().backward()
    assert expanded.extra_gate.weight.grad is not None
    assert torch.count_nonzero(expanded.extra_gate.weight.grad) > 0
    assert expanded.extra_up.weight.grad is not None
    assert torch.count_nonzero(expanded.extra_up.weight.grad) > 0


def test_minigpt4_vqa_uses_explicit_vision_prompt_contract():
    source = open("easyeditor/trainer/blip2_models/mini_gpt4.py", encoding="utf-8").read()
    assert "###Human: <Img><ImageHere></Img> " in source
    assert "targets[i, :prompt_len] = -100" in source
