import torch
import torch.nn as nn

from easyeditor.trainer.algs.MEND import _FunctionalModel
from easyeditor.trainer.BaseTrainer import _main_optimizer_parameters


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.proj(x)


def test_functional_model_preserves_fast_weight_meta_gradient():
    model = TinyModel()
    original = model.proj.weight.detach().clone()
    scale = nn.Parameter(torch.tensor(0.25))
    transformed_gradient = torch.ones_like(model.proj.weight) * 2
    fast_weight = model.proj.weight + scale * transformed_gradient
    edited = _FunctionalModel(model, {"proj.weight": fast_weight})

    loss = edited(torch.ones(1, 3)).sum()
    (meta_grad,) = torch.autograd.grad(loss, [scale])

    assert torch.isfinite(meta_grad)
    assert meta_grad.abs() > 0
    assert torch.equal(model.proj.weight, original)


def test_functional_model_reuses_fast_weights_across_forwards():
    model = TinyModel()
    scale = nn.Parameter(torch.tensor(0.25))
    fast_weight = model.proj.weight + scale * torch.ones_like(model.proj.weight)
    edited = _FunctionalModel(model, {"proj.weight": fast_weight})

    loss = edited(torch.ones(1, 3)).sum() + edited(torch.full((1, 3), 2.0)).sum()
    (meta_grad,) = torch.autograd.grad(loss, [scale])

    assert torch.isfinite(meta_grad)
    assert meta_grad.abs() > 0


class TinyEditor(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Linear(2, 2)
        self.edit_lrs = nn.Parameter(torch.ones(2))

    def outer_parameters(self):
        return list(self.transform.parameters()) + [self.edit_lrs]


def test_main_optimizer_excludes_separately_optimized_edit_lrs():
    editor = TinyEditor()

    main = list(_main_optimizer_parameters(editor))

    assert id(editor.edit_lrs) not in {id(p) for p in main}
    assert {id(p) for p in main} == {
        id(p) for p in editor.transform.parameters()
    }
