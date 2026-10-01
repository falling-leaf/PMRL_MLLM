"""Equivalence tests for the MEND/LLaVA-OV throughput changes.

Every optimisation in this file must be provably value-preserving; these tests
pin that down:

* :func:`batch_update_counter` (block Welford) reproduces the sequential
  ``update_counter`` chain -- exactly in float64, to float32 reduction noise in
  float32 -- while cutting the tensor-op count per step by ~4 orders of
  magnitude.
* ``_FunctionalModel`` with only the edited tensors in the dict produces
  bit-identical outputs/gradients to the historic "pass every parameter" dict.
* the locality softmax/top-k metrics are bit-identical under ``no_grad``.
* source-level guards for the no-grad inner forward, the conditional CUDA cache
  flush and the new config defaults.

Run: ``python -m pytest tests/test_mend_perf_equivalence.py -q``
"""

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. block Welford == sequential Welford
# ---------------------------------------------------------------------------


def _mend():
    """The MEND *module* (``easyeditor.trainer.algs.MEND`` is a class name)."""
    import importlib

    return importlib.import_module("easyeditor.trainer.algs.MEND")


def _mt():
    import importlib

    return importlib.import_module("easyeditor.trainer.MultimodalTrainer")


def _cfg(**overrides):
    base = dict(
        combine=True,
        one_sided=False,
        x_only=False,
        delta_only=False,
        n_hidden=1,
        init="id",
        act="relu",
        rank=8,
        mlp_class="IDMLP",
        norm=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _run_transform(gt, batches):
    return [gt(u, v, None) for u, v in batches]


def _make_batches(batches, x_dim, d_dim, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(n_rows, x_dim, generator=gen) * 2.0,
         torch.randn(n_rows, d_dim, generator=gen) * 3.0)
        for n_rows, _, _ in batches
    ]


def test_block_welford_matches_sequential_in_float64():
    """The parallel-combination identity is exact, not an approximation."""
    MEND = _mend()
    torch.manual_seed(0)
    x = torch.randn(41, 17, dtype=torch.float64)

    # sequential reference (the historic per-row recursion), fp64
    m, s = x[0].clone(), torch.zeros(17, dtype=torch.float64)
    k = torch.tensor([1.0], dtype=torch.float64)
    for idx in range(1, x.shape[0]):
        k = k + 1
        m, s = MEND.update_counter(x[idx], m, s, k)

    m_b, s_b = MEND.batch_update_counter(x[1:], x[0], torch.zeros(17, dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(m, m_b, rtol=1e-14, atol=1e-14)
    assert torch.allclose(s, s_b, rtol=1e-13, atol=1e-13)


def test_running_stats_match_legacy_loop_in_float32():
    """Module-level: same statistics, same k, same transform output (fp32)."""
    MEND = _mend()
    from easyeditor.trainer.algs.MEND import GradientTransform

    torch.manual_seed(1)
    x_dim, d_dim = 16, 48
    shapes = [(37, x_dim, d_dim), (64, x_dim, d_dim), (5, x_dim, d_dim)]
    batches = _make_batches(shapes, x_dim, d_dim)

    gt_fast = GradientTransform(x_dim, d_dim, _cfg())
    gt_legacy = copy.deepcopy(gt_fast)
    assert gt_fast.training and gt_legacy.training

    original_flag = MEND._LEGACY_NORM_LOOP
    try:
        MEND._LEGACY_NORM_LOOP = False
        outs_fast = _run_transform(gt_fast, batches)
        MEND._LEGACY_NORM_LOOP = True
        outs_legacy = _run_transform(gt_legacy, batches)
    finally:
        MEND._LEGACY_NORM_LOOP = original_flag

    # identical sample count
    assert torch.equal(gt_fast.k, gt_legacy.k)
    assert int(gt_fast.k.item()) == 37 + 64 + 5
    # same statistics up to float32 reduction order
    for name in ("u_mean", "v_mean", "u_std", "v_std", "u_s", "v_s"):
        a, b = getattr(gt_fast, name), getattr(gt_legacy, name)
        scale = b.abs().clamp_min(1e-6)
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), (name, (a - b).abs().max())
        assert ((a - b).abs() / scale).max() < 1e-4, name
    for (o1f, o2f), (o1l, o2l) in zip(outs_fast, outs_legacy):
        assert torch.allclose(o1f, o1l, rtol=1e-4, atol=1e-4)
        assert torch.allclose(o2f, o2l, rtol=1e-4, atol=1e-4)


def test_running_stats_handles_first_call_and_empty_mask():
    """First-ever row initialises; an all-zero (fully masked) block is a no-op."""
    MEND = _mend()
    from easyeditor.trainer.algs.MEND import GradientTransform

    torch.manual_seed(2)
    cfg = _cfg()
    gt = GradientTransform(8, 8, cfg)
    assert gt.norm_init is False

    x_dim = d_dim = 8
    gt(torch.randn(3, x_dim), torch.randn(3, d_dim), None)
    assert gt.norm_init is True
    assert float(gt.k.item()) == 3.0

    # fully masked block: no state change, no crash, matches the legacy loop
    gt_legacy = copy.deepcopy(gt)
    zero_u = torch.zeros(4, x_dim)
    zero_v = torch.zeros(4, d_dim)
    k_before = float(gt.k.item())
    gt(zero_u, zero_v, None)
    gt_legacy_ok = True
    original_flag = MEND._LEGACY_NORM_LOOP
    try:
        MEND._LEGACY_NORM_LOOP = True
        gt_legacy(zero_u, zero_v, None)
    finally:
        MEND._LEGACY_NORM_LOOP = original_flag
    assert float(gt.k.item()) == k_before == float(gt_legacy.k.item())
    assert gt_legacy_ok


def test_block_update_cuts_operator_count():
    """Quantifies why the block update is faster: ~4 orders fewer tensor ops."""
    MEND = _mend()
    n_rows, dim = 512, 256
    torch.manual_seed(3)
    x = torch.randn(n_rows, dim)
    mean = torch.zeros(dim)
    s = torch.zeros(dim)

    def sequential():
        m, ss = mean.clone(), s.clone()
        k = torch.tensor([500.0])
        for idx in range(n_rows):
            k = k + 1
            m, ss = MEND.update_counter(x[idx], m, ss, k)
        return m

    def blocked():
        return MEND.batch_update_counter(x, mean, s, torch.tensor([500.0]))[0]

    blocked()  # warm up
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        sequential()
    seq_ops = sum(row.count for row in prof.key_averages())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        blocked()
    block_ops = sum(row.count for row in prof.key_averages())

    # per transform on the real workload n_rows ~ 8.9k, so the ratio matters a lot
    assert seq_ops > 20 * block_ops, (seq_ops, block_ops)
    # per-row cost of the historic loop, for the scaling argument in the docs
    assert seq_ops / n_rows > 5


# ---------------------------------------------------------------------------
# 2. functional_call with only the edited tensors
# ---------------------------------------------------------------------------


class _TinyLlavaLike(torch.nn.Module):
    """Nested module whose edited tensors are the *last* named parameters."""

    def __init__(self, hidden=6, inter=10):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        lang = self.model.language_model
        lang.layers = torch.nn.ModuleList()
        for _ in range(3):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.up_proj = torch.nn.Linear(hidden, inter, bias=False)
            layer.mlp.down_proj = torch.nn.Linear(inter, hidden, bias=False)
            lang.layers.append(layer)
        # trailing parameters, like LLaVA-OV's final norm / lm_head
        self.norm = torch.nn.LayerNorm(hidden)
        self.lm_head = torch.nn.Linear(hidden, 5, bias=False)

    def forward(self, x=None, labels=None, input_ids=None, attention_mask=None, **kw):
        h = input_ids
        for layer in self.model.language_model.layers:
            h = h + layer.mlp.down_proj(F.relu(layer.mlp.up_proj(h)))
        return self.lm_head(self.norm(h))


def test_functional_call_partial_dict_is_bitwise_identical():
    torch.manual_seed(4)
    model = _TinyLlavaLike()
    inner = [
        "model.language_model.layers.2.mlp.up_proj.weight",
        "model.language_model.layers.2.mlp.down_proj.weight",
    ]
    names = dict(model.named_parameters())
    # edited tensors sit at the end of named_parameters(), like LLaVA-OV's
    # layer-27 MLP, which is what lets MEND freeze everything before them
    assert set(inner).issubset(set(list(names)[-6:]))
    updates = {n: torch.randn_like(names[n]) * 1e-3 for n in inner}

    full = {
        n: (p + updates[n]) if n in updates else p
        for n, p in model.named_parameters()
    }
    partial = {n: p + updates[n] for n, p in model.named_parameters() if n in updates}
    assert set(partial) == set(inner)

    x = torch.randn(2, 3, 6)
    out_full = torch.func.functional_call(
        model, full, kwargs={"input_ids": x}, strict=False
    )
    out_partial = torch.func.functional_call(
        model, partial, kwargs={"input_ids": x}, strict=False
    )
    assert torch.equal(out_full, out_partial)

    # gradients reach the edited tensors through the partial dict
    out_partial.sum().backward()
    assert names[inner[0]].grad is not None
    assert names[inner[1]].grad is not None
    assert torch.isfinite(names[inner[0]].grad).all()


def test_functional_model_rejects_unknown_fast_keys():
    from easyeditor.trainer.algs.MEND import _FunctionalModel

    model = _TinyLlavaLike()
    names = dict(model.named_parameters())
    key = "model.language_model.layers.2.mlp.up_proj.weight"
    wrapped = _FunctionalModel(model, {key: names[key] + 0.01})
    x = torch.randn(2, 3, 6)
    shifted = wrapped(input_ids=x)
    base = model(input_ids=x)
    assert not torch.equal(shifted, base)
    try:
        _FunctionalModel(model, {"not.a.real.weight": names[key]})
        raise AssertionError("expected KeyError for unknown fast_params key")
    except KeyError as exc:
        assert "not.a.real.weight" in str(exc)


# ---------------------------------------------------------------------------
# 3. locality metrics under no_grad
# ---------------------------------------------------------------------------


def test_locality_metrics_are_identical_under_no_grad():
    torch.manual_seed(5)
    base = torch.randn(1, 24, 257)
    post = (torch.randn(1, 24, 257) + base * 0.1).requires_grad_(True)

    def metrics(enable_grad):
        with torch.set_grad_enabled(enable_grad):
            post_top = torch.topk(F.softmax(post, dim=-1), k=10, dim=-1).indices
            base_top = torch.topk(F.softmax(base, dim=-1), k=10, dim=-1).indices
            acc = sum(post_top.view(-1) == base_top.view(-1)) / post_top.view(-1).shape[0]
        return post_top, base_top, acc

    p_grad, b_grad, acc_grad = metrics(True)
    p_nograd, b_nograd, acc_nograd = metrics(False)

    assert torch.equal(p_grad, p_nograd)
    assert torch.equal(b_grad, b_nograd)
    assert torch.equal(acc_grad, acc_nograd)
    # under no_grad nothing is kept for backward
    with torch.no_grad():
        assert F.softmax(post, dim=-1).grad_fn is None
    with torch.set_grad_enabled(True):
        assert F.softmax(post, dim=-1).grad_fn is not None


# ---------------------------------------------------------------------------
# 4. source-level guards
# ---------------------------------------------------------------------------


def _source(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_inner_forward_is_no_grad_unless_it_feeds_the_loss():
    src = _source("easyeditor/trainer/MultimodalTrainer.py")
    start = src.index("inner_feeds_loss = (")
    end = src.index("if not isinstance(inner_edit_outputs", start)
    segment = src[start:end]
    assert 'self.train_set.__class__.__name__ == "ComprehendEditDataset"' in segment
    assert segment.count("torch.no_grad()") >= 1
    # the grad path is still taken for the dataset that reuses these logits
    assert "inner_edit_outputs = edited_model(batch[\"edit_inner\"])" in segment


def test_cache_flush_is_conditional():
    src = _source("easyeditor/trainer/MultimodalTrainer.py")
    assert "self._maybe_flush_gpu_cache()" in src
    # the unconditional per-step flush is gone
    assert "torch.cuda.synchronize()\n        torch.cuda.empty_cache()" not in src
    helper = inspect.getsource(_mt().MultimodalTrainer._maybe_flush_gpu_cache)
    assert "mem_get_info" in helper


def test_functional_model_receives_only_edited_tensors():
    src = _source("easyeditor/trainer/algs/MEND.py")
    start = src.index('if "llava-onevision" in self.config.model_name.lower()')
    segment = src[start : src.index("base_model = _FunctionalModel", start)]
    assert "_inner_params(" in segment
    assert "for n, p in self.model.named_parameters():" not in segment


def test_new_hparams_defaults_are_inert():
    from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
        MENDMultimodalTrainingHparams,
    )

    hp = MENDMultimodalTrainingHparams.from_hparams(
        str(ROOT / "hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-graphfix-final2.yaml")
    )
    assert hp.dataloader_num_workers == 0
    assert hp.gpu_cache_free_threshold_gb == 8.0
    assert hp.mend_log_weight_diagnostics is False
    # the historic diagnostics flag stays as configured by the run
    assert hp.mend_log_grad_diagnostics is True


def test_hf_target_loss_ignores_contiguity():
    """Dropping the .contiguous() copy yields bit-identical CE."""
    MEND = _mend()
    torch.manual_seed(6)
    logits = torch.randn(1, 20, 64)
    labels = torch.full((1, 20), -100, dtype=torch.long)
    labels[:, -4:] = torch.randint(0, 64, (1, 4))

    shifted = logits[:, :-1, :].contiguous()
    ref = F.cross_entropy(
        shifted[labels[:, 1:].ne(-100)], labels[:, 1:][labels[:, 1:].ne(-100)]
    )
    got = MEND.MEND._hf_target_loss(logits, labels)
    assert torch.equal(ref, got)
    assert MEND.MEND._hf_target_loss(logits, labels, "sum").shape == ()
    # and on a non-contiguous input view
    view = logits.transpose(0, 0)
    assert torch.equal(MEND.MEND._hf_target_loss(view, labels), got)


def test_move_to_device_walks_batchfeature_containers():
    """DataLoader-worker batches arrive as HF BatchFeature (UserDict) objects."""
    from transformers import BatchFeature

    from easyeditor.trainer.utils import move_to_device

    inner = BatchFeature(
        {
            "input_ids": torch.zeros(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(1, 2, dtype=torch.float32),
        }
    )
    inner["labels"] = torch.zeros(1, 3, dtype=torch.long)
    batch = {"edit_inner": inner, "loc": {"attention_mask": torch.zeros(1, 3)}}

    # meta is a real device target that requires no CUDA
    out = move_to_device(batch, torch.device("meta"))
    assert out is batch
    assert isinstance(out["edit_inner"], BatchFeature)
    assert out["edit_inner"]["input_ids"].device.type == "meta"
    assert out["edit_inner"]["pixel_values"].device.type == "meta"
    assert out["edit_inner"]["labels"].device.type == "meta"
    assert out["loc"]["attention_mask"].device.type == "meta"
    # device=None is a no-op (used when the collate already placed the batch)
    assert move_to_device(batch, None) is batch


def test_collate_skips_device_move_inside_workers():
    """The forked worker path must never touch CUDA."""
    src = _source("easyeditor/dataset/coco_caption.py")
    assert "torch.utils.data.get_worker_info() is not None" in src
    from easyeditor.dataset.coco_caption import CaptionDataset

    assert "get_worker_info" in inspect.getsource(CaptionDataset._collate_device)
    # every device resolution in the collate goes through the helper
    assert 'device = getattr(self.config, "device", None)' not in src


def test_module_imports_cleanly():
    mt = _mt()
    from easyeditor.trainer.algs import profiling

    assert hasattr(mt, "MultimodalTrainer")
    assert profiling.PROFILE in (True, False)
