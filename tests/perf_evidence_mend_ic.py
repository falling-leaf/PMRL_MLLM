"""CPU-side evidence for the MEND/LLaVA-OV IC step-speed analysis.

Three independent measurements, all runnable without a GPU:

1. Data pipeline cost: dataset construction, per-step collate time and the real
   per-sub-batch sequence lengths for the IC configuration.
2. The gradient-transform normalisation loop: tensor-op counts of the historic
   per-row Welford loop vs the block update, extrapolated to the measured row
   counts, plus the fp32 deviation between the two implementations.
3. A summary of the per-step model-forward budget implied by those numbers.

Note: this box has a 2 GiB cgroup memory cap shared with other tenants, so the
measurements are taken on small tensors and scaled (op count per row is
independent of the feature dim).  The ``data`` section still needs ~900 MB of
free cgroup memory (torch + transformers + the LLaVA-OV processor + the collated
batch); when other processes hold the rest of the cap it is SIGKILLed with exit
137 before printing anything beyond its header -- free the cap (or run on an
idle box) and retry rather than assuming a bug in the script.

Run: ``PYTHONPATH=. python tests/perf_evidence_mend_ic.py``
"""

import gc
import os
import time

import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CFG = "hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-graphfix-final2.yaml"


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def data_pipeline():
    section("1. DATA PIPELINE (CPU)")
    from easyeditor import CaptionDataset
    from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
        MENDMultimodalTrainingHparams,
    )

    hp = MENDMultimodalTrainingHparams.from_hparams(CFG)
    hp.device = "cpu"
    hp.dtype = torch.bfloat16

    t0 = time.time()
    ds = CaptionDataset(
        "/root/MMEdit/editing-data/caption/caption_train_edit.json", config=hp, size=4
    )
    print(f"dataset_init_seconds      : {time.time() - t0:.2f} (3 Image.open per record)")

    times = []
    for _ in range(3):
        t0 = time.time()
        batch = ds.collate_fn([ds[0]])
        times.append(time.time() - t0)
    print(f"collate_seconds (per step): {min(times):.3f} - {max(times):.3f}")

    image_token_id = getattr(ds.tok, "image_token_id", None)
    seq = {}
    print(f"{'sub-batch':<20s} {'seq':>7s} {'image_tok':>10s} {'supervised':>11s} {'tiles':>6s}")
    for key in ["edit_inner", "edit_outer", "edit_outer_image", "loc", "loc_image"]:
        ids = batch[key]["input_ids"]
        n_img = int((ids == image_token_id).sum()) if image_token_id is not None else 0
        n_sup = int((batch[key]["labels"] != -100).sum())
        tiles = batch[key].get("pixel_values")
        n_tiles = int(tiles.shape[1]) if torch.is_tensor(tiles) else 0
        print(f"{key:<20s} {ids.shape[1]:7d} {n_img:10d} {n_sup:11d} {n_tiles:6d}")
        seq[key] = int(ids.shape[1])
    print(f"cond seq: {int(batch['cond']['input_ids'].shape[1])}")

    del batch, ds
    gc.collect()
    return seq


def _cfg():
    from types import SimpleNamespace

    return SimpleNamespace(
        combine=True, one_sided=False, x_only=False, delta_only=False,
        n_hidden=1, init="id", act="relu", rank=8, mlp_class="IDMLP", norm=True,
    )


def _count_ops(fn, n_rows):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        fn()
    return sum(r.count for r in prof.key_averages())


def transform_norm_loop(seq):
    section("2. GRADIENT-TRANSFORM NORMALISATION LOOP (CPU, op counting)")
    import copy
    import importlib

    MEND = importlib.import_module("easyeditor.trainer.algs.MEND")

    n_views = 1 + 2  # edit forward + num_rephrase LAP variant views (cfg: 2)
    real_rows = n_views * seq["edit_inner"]
    print(f"captured rows per transform: {n_views} views x {seq['edit_inner']} tokens = {real_rows}")

    dim = 32
    mean = torch.zeros(dim)
    s = torch.zeros(dim)
    per_row = None
    block_ops = None
    for n_rows in (512, 2048):
        x = torch.randn(n_rows, dim)

        def sequential():
            m, ss = mean.clone(), s.clone()
            k = torch.tensor([500.0])
            for idx in range(n_rows):
                k = k + 1
                m, ss = MEND.update_counter(x[idx], m, ss, k)
            return m

        def blocked():
            return MEND.batch_update_counter(x, mean, s, torch.tensor([500.0]))[0]

        blocked()
        seq_ops = _count_ops(sequential, n_rows)
        b_ops = _count_ops(blocked, n_rows)
        print(f"n_rows={n_rows:5d}: sequential={seq_ops:7d} ops  block={b_ops:5d} ops")
        per_row = seq_ops / n_rows
        block_ops = b_ops
        del x
        gc.collect()

    ops_per_row = per_row - 1  # the per-row k += 1 is shared by both counters
    total_ops = (ops_per_row * 2 + 1) * 2 * real_rows  # 2 counters, 2 transforms
    print(f"per-row cost of the historic loop: {per_row:.2f} tensor ops per row per counter")
    print("the loop updates a u- and a v-counter for each of the 2 shared transforms:")
    print(f"  per step  : {total_ops:,.0f} tensor ops  (block update: ~{block_ops * 4})")
    for us in (3, 5):
        print(f"  ~{us}us per CUDA launch -> {total_ops * us * 1e-6:.2f}s/step just in launches")

    torch.manual_seed(0)
    x32 = torch.randn(2048, 256) * 2
    gt_fast = MEND.GradientTransform(256, 256, _cfg())
    gt_legacy = copy.deepcopy(gt_fast)
    old = MEND._LEGACY_NORM_LOOP
    try:
        MEND._LEGACY_NORM_LOOP = False
        gt_fast(x32, x32, None)
        MEND._LEGACY_NORM_LOOP = True
        gt_legacy(x32, x32, None)
    finally:
        MEND._LEGACY_NORM_LOOP = old
    dev = (gt_fast.u_mean - gt_legacy.u_mean).abs().max().item()
    rel = (
        (gt_fast.u_mean - gt_legacy.u_mean).abs() / gt_legacy.u_mean.abs().clamp_min(1e-6)
    ).max().item()
    std_dev = (gt_fast.u_std - gt_legacy.u_std).abs().max().item()
    print(f"fp32 stats deviation       : max|dmean|={dev:.3e} max_rel={rel:.3e} max|dstd|={std_dev:.3e}")
    print(f"k identical                : {torch.equal(gt_fast.k, gt_legacy.k)}")
    del x32, gt_fast, gt_legacy
    gc.collect()


def forward_budget(seq):
    section("3. PER-STEP MODEL FORWARD BUDGET (derived)")
    n_params = 7.5e9
    print("long-sequence forwards per step:")
    print("  pre-edit : loc(text-only) + loc_image                 -> 2")
    print("  edit()   : edit + LAP probe + num_rephrase variants    -> 4")
    print("  post-edit: edit_outer, edit_inner, edit_outer_image, loc, loc_image -> 5")
    print("backward passes: 3 inside edit() (edit + 2 variants), then the meta loss")
    for k, v in seq.items():
        print(f"  seq[{k}] = {v}")
    L = seq["edit_inner"]
    print(f"2*N*L for a {L}-token forward at N=7.5e9: {2 * n_params * L / 1e12:.1f} TFLOP")
    print("=> ~12 forwards + 6-7 forward-equivalents of backward per step; 7.3 s/step")
    print("   therefore implies ~0.4 s per forward: the step is dominated by real")
    print("   7B GEMM compute, so the removable overhead is the python/launch side.")


if __name__ == "__main__":
    # This container caps the process at ~2 GiB, and holding the LLaVA-OV
    # processor (data pipeline section) while running the profiler exceeds it,
    # so the sections are selectable:  ``... ops`` (default), ``... data``,
    # ``... all`` (may be killed by the memory cap).
    import sys

    what = sys.argv[1] if len(sys.argv) > 1 else "ops"
    if what == "data":
        forward_budget(data_pipeline())
    elif what == "all":
        seq = data_pipeline()
        transform_norm_loop(seq)
        forward_budget(seq)
    else:
        # sequence lengths measured by ``... data`` on the IC configuration
        seq = {
            "edit_inner": 2964,
            "edit_outer": 2970,
            "edit_outer_image": 3720,
            "loc": 31,
            "loc_image": 2953,
        }
        transform_norm_loop(seq)
        forward_budget(seq)
