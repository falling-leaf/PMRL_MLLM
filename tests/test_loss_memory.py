from types import SimpleNamespace

import torch

from easyeditor.trainer.losses import masked_log_probs


class NoWholeLogitCopyTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor):
        return torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)

    def clone(self, *args, **kwargs):
        raise AssertionError("full logits tensor must not be cloned")


def test_masked_log_probs_does_not_clone_full_logits():
    logits = NoWholeLogitCopyTensor(torch.randn(1, 4, 7, dtype=torch.bfloat16))
    labels = torch.tensor([[-100, 1, 2, 3]])
    config = SimpleNamespace(model_class="llava-onevision")

    result = masked_log_probs(config, logits, labels, shift=True, multimodal=True)

    assert torch.isfinite(result["nll"])
    assert torch.isfinite(result["acc"])
