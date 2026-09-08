import torch
from torch import nn

from easyeditor.models.unike_simplified import ExpandedSwiGLUMLP


def test_expanded_swiglu_intrinsic_parameters_use_float32_master_weights():
    original = nn.Module()
    original.gate_proj = nn.Linear(4, 6, bias=False).half()
    original.up_proj = nn.Linear(4, 6, bias=False).half()
    original.down_proj = nn.Linear(6, 4, bias=False).half()
    expanded = ExpandedSwiGLUMLP(original, add_neuron_num=10)
    assert expanded.extra_gate.weight.dtype == torch.float32
    assert expanded.extra_up.weight.dtype == torch.float32
    assert expanded.extra_down.weight.dtype == torch.float32
