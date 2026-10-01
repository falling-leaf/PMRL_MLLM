from types import SimpleNamespace

import torch

from easyeditor.trainer.losses import kl_loc_loss, masked_log_probs


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


def test_chunked_sequence_kl_matches_reference_value_and_gradient():
    torch.manual_seed(7)
    pre = torch.randn(5, 11, dtype=torch.float32)
    post = torch.randn(5, 11, dtype=torch.float32, requires_grad=True)
    mask = torch.tensor([True, False, True, True, False])

    result = kl_loc_loss(pre, post, mask=mask, chunk_size=2)
    result.backward()
    actual_grad = post.grad.detach().clone()

    with torch.no_grad():
        p = pre[mask].softmax(-1)
        q = post.detach()[mask]
        expected = (p * (p.log() - q.log_softmax(-1))).sum(-1).mean()
        expected_grad = torch.zeros_like(post)
        expected_grad[mask] = (q.softmax(-1) - p) / mask.sum()

    assert torch.allclose(result.detach(), expected, rtol=1e-6, atol=1e-6)
    assert torch.allclose(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)


def test_chunked_sequence_kl_rejects_empty_mask():
    pre = torch.randn(2, 5)
    post = torch.randn(2, 5, requires_grad=True)
    try:
        kl_loc_loss(pre, post, mask=torch.zeros(2, dtype=torch.bool), chunk_size=1)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty KL mask must be rejected")
