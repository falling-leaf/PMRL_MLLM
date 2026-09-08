"""Focused unit tests for the isolated UniKE-BLIP2 mechanism."""
from types import SimpleNamespace

import torch
from torch import nn

from easyeditor.models.unike_blip2.unike_blip2_main import _FeatureShift


def test_feature_shift_is_norm_preserving_and_finite():
    module = _FeatureShift(
        memory=torch.tensor([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]]),
        top_k=2,
        scale=1.0,
        semantic_gate=True,
    )
    original = torch.tensor([[[1.0, 0.2], [0.3, 1.0]]])
    shifted = module(nn.Identity(), (), original)
    assert torch.isfinite(shifted).all()
    assert torch.allclose(shifted.norm(dim=-1), original.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_feature_shift_handles_memory_shorter_than_requested_topk():
    module = _FeatureShift(torch.tensor([[1.0, 0.0]]), top_k=40, scale=1.0, semantic_gate=False)
    output = module(nn.Identity(), (), torch.tensor([[[0.0, 2.0]]]))
    assert output.shape == (1, 1, 2)
    assert torch.isfinite(output).all()
