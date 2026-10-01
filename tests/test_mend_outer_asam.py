import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(fullname, rel, sys_modules=None):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    if sys_modules is not None:
        sys_modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def _load_outer_stack():
    saved = {}
    names = [
        "easyeditor",
        "easyeditor.util",
        "easyeditor.util.pmrl_utils",
        "easyeditor.trainer",
        "easyeditor.trainer.algs",
        "easyeditor.trainer.algs.outer_asam",
        "easyeditor.trainer.utils",
    ]
    for name in names:
        if name in sys.modules:
            saved[name] = sys.modules[name]

    easyeditor = types.ModuleType("easyeditor")
    util = types.ModuleType("easyeditor.util")
    trainer = types.ModuleType("easyeditor.trainer")
    algs = types.ModuleType("easyeditor.trainer.algs")
    easyeditor.util = util
    easyeditor.trainer = trainer
    trainer.algs = algs
    sys.modules["easyeditor"] = easyeditor
    sys.modules["easyeditor.util"] = util
    sys.modules["easyeditor.trainer"] = trainer
    sys.modules["easyeditor.trainer.algs"] = algs
    pmrl = _load("easyeditor.util.pmrl_utils", "easyeditor/util/pmrl_utils.py", sys.modules)
    util.pmrl_utils = pmrl
    utils = _load("easyeditor.trainer.utils", "easyeditor/trainer/utils.py", sys.modules)
    trainer.utils = utils
    outer = _load(
        "easyeditor.trainer.algs.outer_asam",
        "easyeditor/trainer/algs/outer_asam.py",
        sys.modules,
    )
    return outer, utils, saved


outer, utils, _saved_modules = _load_outer_stack()


def test_asam_delta_roundtrip_restores_weights():
    torch.manual_seed(0)
    params = [nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
    before = [p.detach().clone() for p in params]
    grads = [torch.randn_like(p) for p in params]
    from easyeditor.util.pmrl_utils import apply_asam_deltas, asam_parameter_deltas

    deltas = asam_parameter_deltas(params, grads, epsilon=0.002, rho=0.1)
    apply_asam_deltas(params, deltas, sign=1.0)
    assert any(not torch.equal(p, b) for p, b in zip(params, before))
    apply_asam_deltas(params, deltas, sign=-1.0)
    for parameter, original in zip(params, before):
        assert torch.equal(parameter, original)


def test_outer_asam_replace_keeps_second_pass_grad():
    torch.manual_seed(1)
    params = [nn.Parameter(torch.ones(3))]
    first = [torch.full_like(params[0], 2.0)]
    params[0].grad = first[0].clone()

    def second_pass():
        params[0].grad = torch.full_like(params[0], 7.0)

    result = outer.apply_outer_asam_grads(
        params, first, second_pass, epsilon=0.01, replace=True
    )
    assert result["skipped"] is False
    assert torch.allclose(params[0].grad, torch.full_like(params[0], 7.0))
    assert torch.equal(params[0], torch.ones(3))


def test_outer_asam_nonfinite_second_pass_restores_first_grad():
    torch.manual_seed(2)
    params = [nn.Parameter(torch.ones(3))]
    first = [torch.full_like(params[0], 1.5)]
    params[0].grad = first[0].clone()

    def second_pass():
        params[0].grad = torch.full_like(params[0], float("nan"))

    result = outer.apply_outer_asam_grads(
        params, first, second_pass, epsilon=0.01, replace=True
    )
    assert result["skipped"] is True
    assert torch.allclose(params[0].grad, first[0])
    assert torch.equal(params[0], torch.ones(3))


def test_outer_asam_exclude_skips_neighborhood_for_edit_lrs():
    torch.manual_seed(0)
    hyper = nn.Parameter(torch.ones(8) * 0.5)
    lrs = nn.Parameter(torch.tensor([1e-4, 1e-4]))
    first_h = torch.ones_like(hyper)
    first_l = torch.ones_like(lrs)
    from easyeditor.util.pmrl_utils import asam_parameter_deltas

    joint = asam_parameter_deltas([hyper, lrs], [first_h, first_l], 0.002, rho=0.1)
    excluded = asam_parameter_deltas([hyper, lrs], [first_h, None], 0.002, rho=0.1)
    assert excluded[1] is None
    assert joint[1] is not None
    # rho=0.1 floors |edit_lrs|=1e-4, so the *relative* SAM step on lrs
    # dwarfs the hypernetwork step even though ||T g|| is still MLP-heavy.
    rel_lrs = (joint[1] / lrs.detach()).abs().max()
    rel_hyper = (joint[0] / hyper.detach()).abs().max()
    assert rel_lrs > 10 * rel_hyper

    before_h = hyper.detach().clone()
    before_l = lrs.detach().clone()

    def second_pass():
        hyper.grad = torch.full_like(hyper, 3.0)
        lrs.grad = torch.full_like(lrs, 4.0)

    result = outer.apply_outer_asam_grads(
        [hyper, lrs],
        [first_h, first_l],
        second_pass,
        epsilon=0.002,
        rho=0.1,
        exclude=[lrs],
    )
    assert result["skipped"] is False
    assert torch.equal(hyper, before_h)
    assert torch.equal(lrs, before_l)
    assert torch.allclose(hyper.grad, torch.full_like(hyper, 3.0))
    assert torch.allclose(lrs.grad, torch.full_like(lrs, 4.0))


def test_zero_rho_leaves_identity_u_unperturbed():
    u = nn.Parameter(torch.zeros(6, 4))
    v = nn.Parameter(torch.ones(4, 6))
    grads = [torch.ones_like(u), torch.ones_like(v)]
    from easyeditor.util.pmrl_utils import asam_parameter_deltas

    floored = asam_parameter_deltas([u, v], grads, 0.002, rho=0.1)
    kwon = asam_parameter_deltas([u, v], grads, 0.002, rho=0.0)
    assert floored[0] is not None and floored[0].abs().sum() > 0
    assert kwon[0] is not None
    assert torch.equal(kwon[0], torch.zeros_like(u))


def test_outer_asam_exception_does_not_leave_perturbed_weights():
    params = [nn.Parameter(torch.ones(2))]
    first = [torch.ones_like(params[0])]

    def second_pass():
        raise RuntimeError("boom")

    result = outer.apply_outer_asam_grads(params, first, second_pass, epsilon=0.01)
    assert result["skipped"] is True
    assert torch.allclose(params[0], torch.ones(2))
    assert torch.equal(params[0].grad, first[0])


def test_loc_floored_score_rejects_collapsed_locality():
    stats = {
        "edit/acc_val": 0.9,
        "image_rephrase/acc_val": 0.9,
        "loc/acc_val": 0.01,
        "image_loc/acc_val": 0.7,
    }
    assert utils.loc_floored_score(stats, t_floor=0.97, m_floor=0.71) == -1.0
    stats["loc/acc_val"] = 0.98
    score = utils.loc_floored_score(stats, t_floor=0.97, m_floor=0.71)
    assert score > 0
    assert abs(score - (0.9 + 0.9 + 0.98 + 0.7)) < 1e-12


def test_vqa_asam_v2_yaml_excludes_edit_lrs_and_drops_zero_floor():
    with open(
        ROOT / "hparams/TRAINING/MEND/llavaov-7b-vqa-asam-outer-earlystop-v2.yaml",
        encoding="utf-8",
    ) as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["using_asam"] is True
    assert cfg["using_extra"] is False
    assert cfg["asam_exclude_edit_lrs"] is True
    assert cfg["asam_scale_rho"] == 0.0
    assert cfg["overfit_stop"] is True
    assert cfg["final_eval"] is False
    assert cfg["results_dir"] == "./results/MEND_LLAVAOV_VQA_ASAM_OUTER_EARLYSTOP_V2"


def test_ic_and_vqa_asam_yamls_are_outer_only():
    for path in (
        ROOT / "hparams/TRAINING/MEND/llavaov-7b-ic-asam-outer.yaml",
        ROOT / "hparams/TRAINING/MEND/llavaov-7b-vqa-asam-outer.yaml",
    ):
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        assert cfg["using_asam"] is True
        assert cfg["using_extra"] is False
        assert cfg["using_lap"] is False
        assert cfg["using_pmrl"] is False
        assert cfg["asam_replace"] is True
        assert cfg["early_stop_key"] == "acc/loc_floored_val"
        assert cfg["model_save_pt"] == 2000
        assert cfg["norm"] is True
        assert cfg["baseline_t_loc"] is not None
        assert cfg["baseline_m_loc"] is not None


def test_functional_model_source_rejects_unknown_keys():
    source = (ROOT / "easyeditor/trainer/algs/MEND.py").read_text(encoding="utf-8")
    start = source.index("class _FunctionalModel")
    body = source[start : source.index("def update_counter")]
    assert "if missing:" in body
    assert "strict=False) would silently skip the edit" in body
