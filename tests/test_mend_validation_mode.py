from types import SimpleNamespace

import torch
import torch.nn as nn

from easyeditor.trainer.algs.MEND import MEND


def test_mend_edit_accepts_validation_flag():
    # The public signature must distinguish training (LAP enabled) from
    # validation (ordinary edit gradient only).
    assert "training" in MEND.edit.__code__.co_varnames
    assert MEND.edit.__defaults__[-1] is True


def test_validation_probe_is_not_needed_for_plain_loss():
    logits = torch.randn(1, 4, 9, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3]])
    loss = MEND._hf_target_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
