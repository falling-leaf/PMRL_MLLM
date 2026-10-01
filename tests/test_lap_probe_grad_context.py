import torch

from easyeditor.trainer.algs.MEND import MEND


def test_lap_probe_can_build_gradients_during_no_grad_validation():
    probe = torch.randn(2, 3, requires_grad=True)

    with torch.no_grad():
        with torch.enable_grad():
            loss = (probe.square()).sum()
            grad = torch.autograd.grad(loss, probe)[0]

    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
