import torch
from torch import nn

from easyeditor.models.unike_simplified import _OnlineMemoryShift


def test_online_memory_shift_uses_soft_topk_and_preserves_norm():
    shift = _OnlineMemoryShift(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]]),
        scale=0.2,
        top_k=2,
        semantic_gate=True,
    )
    x = torch.tensor([[[1.0, 0.2], [0.3, 1.0]]])
    y = shift(nn.Identity(), (), x)
    assert torch.isfinite(y).all()
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)
    assert shift.top_k == 2
