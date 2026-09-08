from types import SimpleNamespace

import torch
from torch import nn

from easyeditor.models.unike_blip2.unike_blip2_main import _FeatureShift


def test_asam_config_is_explicit_and_parses():
    from easyeditor.models.unike_blip2 import UniKEBLIP2HyperParams
    p = UniKEBLIP2HyperParams.from_hparams("hparams/UniKE/blip2_ic_unike_asam.yaml")
    assert p.using_asam is True
    assert p.asam_epsilon == 0.1
    assert p.asam_weight == 1.0


def test_feature_shift_is_finite_under_asam_style_dtype():
    shift = _FeatureShift(torch.tensor([[1.0, 0.0], [0.8, 0.2]]), 40, 1.0, True)
    x = torch.tensor([[[1.0, 0.2], [0.3, 1.0]]], dtype=torch.float32)
    y = shift(nn.Identity(), (), x)
    assert torch.isfinite(y).all()
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)
