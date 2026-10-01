import os
import tempfile
from types import SimpleNamespace

import torch

from easyeditor.trainer.BaseTrainer import BaseTrainer
from easyeditor.trainer.losses import masked_log_probs


def test_checkpoint_writes_are_atomic(monkeypatch, tmp_path):
    # Exercise the same serialization primitive used by BaseTrainer without
    # constructing the 7B model.
    target = tmp_path / "model"
    payload = {"step": 7, "model": {"x": torch.ones(2)}}
    temp = tmp_path / "model.tmp.test"
    with open(temp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["step"] == 7
    assert not temp.exists()


def test_validation_loss_path_does_not_clone_full_logits():
    class Guarded(torch.Tensor):
        @staticmethod
        def __new__(cls, x):
            return torch.Tensor._make_subclass(cls, x, x.requires_grad)
        def clone(self, *args, **kwargs):
            raise AssertionError("full logits clone detected")

    logits = Guarded(torch.randn(1, 4, 9, dtype=torch.bfloat16))
    labels = torch.tensor([[-100, 1, 2, 3]])
    out = masked_log_probs(
        SimpleNamespace(model_class="llava-onevision"),
        logits,
        labels,
        shift=True,
        multimodal=True,
    )
    assert torch.isfinite(out["nll"])
