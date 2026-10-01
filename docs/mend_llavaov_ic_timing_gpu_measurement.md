# LLaVA-OV MEND + LAP/PMRL (ASAM) IC: GPU step-time measurement and 30 000-step ETA

Machine-readable twin: `results/mend_llavaov_ic_perf_timing.json`
(regenerate with `/root/miniconda3/envs/easyedit/bin/python tests/mend_ic_timing_report.py`).
Parser for the `MEND_PROFILE` tables: `tests/parse_mend_profile_log.py`.

Experiment group: `hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-graphfix-final2.yaml`
(interrupted 09/10 run = `results/MEND_LLAVAOV_LAP_PMRL_IC_GRAPHFIX_FINAL2`).
All runs below were executed 2026-09-12 on the GPU box (RTX 6000D, 85 GB,
torch 2.9.1+cu128, transformers 4.57.1, `easyedit` env python 3.10).

The timing runs use `hparams/.../llavaov-7b-lap-pmrl-ic-graphfix-final2-timing.yaml`,
a byte-identical copy of the frozen config except `results_dir`
(`results/MEND_LLAVAOV_IC_PERF_TIMING_100`), `log_interval: 10`,
`val_interval: 1000000` and `final_eval: false` - no training hyperparameter
was touched. Throughput-only env switches (`PMRL_MEND_*`, `MEND_*`) are the
documented knobs of `run_mend_llavaov_ic_train.py`; the frozen yaml is untouched.

| run | log | setting | steps | exit |
|-----|-----|---------|-------|------|
| A | `run_logs/mend_llavaov_ic_perf_n100_w0.log` | frozen config, `workers=0`, profiled | 100 | 0 |
| C | `run_logs/perf_perf_n40_w4.log` | `PMRL_MEND_NUM_WORKERS=4` | 40 | 0 |
| D | `run_logs/perf_perf_n20_seqnorm.log` | `MEND_NORM_SEQUENTIAL=1` (pre-optimisation loop) | 20 | 0 |
| F | `run_logs/perf_perf_n24_val.log` | frozen config + one validation at step 20 | 24 | 0 |
| G/H | `run_logs/perf_perf_eq_{block,seq}.log` | `log_interval=1`, block vs legacy Welford | 3 | 0 |
| RA/RB | `run_logs/perf_perf_rep_{a,b}.log` | two *identical* invocations (noise floor) | 20 | 0 |

---

## 1. Step time (measured, not estimated)

| run | s/step (steady) | min-max | de-duplicated section sum | residual (collate + optim + logging) |
|-----|-----------------|---------|---------------------------|--------------------------------------|
| A frozen, workers=0 | **5.487** | 5.30-5.70 | 4.832 | 0.655 |
| C frozen, workers=4 | **5.200** | 5.10-5.30 | 4.717 | 0.483 |
| D legacy Welford loop | 7.000 | 7.00 | 6.340 | 0.660 |
| RA / RB (identical repeats) | 5.3 / 5.3 | 5.3 | 4.645 / 4.652 | 0.655 / 0.648 |
| pre-optimisation reference (09/10 `...GRAPHFIX_FINAL2`) | 7.424 | - | - | - |

Run A: 100 steps in 549 s of steady training (steps 11-100: 492 s / 90 steps);
626 s wall clock for the whole process including 67 s startup and 2 s teardown.
The step-time figure comes from the trainer's own step echoes, so it is
independent of `MEND_PROFILE`; the profiler only adds ~34 `cudaSynchronize`
calls per step, which are inside the section timers.

**Same-day A/B for the flagship change** (`MEND_NORM_SEQUENTIAL=1`, same code,
same config, same seed): 5.30 s/step vs 7.00 s/step in the same window index.

## 2. Where the step time goes (run A, ms/step, de-duplicated)

```
edit.loss                    1409.3   (of which lap_probe 697.1, lap_view 664.4,
                                      pmrl 3.2 -> pure CE ~45)
step.meta_backward            735.7
step.fwd_post_edit_outer_image 480.4
step.fwd_post_edit_inner      370.7
edit.fwd_edit                 369.3
step.fwd_post_edit_outer      369.1
step.fwd_post_loc_image       331.4
step.fwd_base_loc_image       329.7
edit.backward                 197.3
step.fwd_base_loc              36.6
step.fwd_post_loc              34.4
edit.einsum                    29.7
step.kl_loc                    23.0
mend.norm_stats                 5.5   <- was 1684.5 with MEND_NORM_SEQUENTIAL=1
```

`edit.loss` contains the LAP probe/views (nested timers), so the report of
`measured_total_s` in the log double counts them; `tests/parse_mend_profile_log.py`
removes the nesting when it builds the table above.

`mend.norm_stats` measures the two `GradientTransform` running-stat updates:
**5.5 ms/step** with the block Welford update vs **1684.5 ms/step** with the
legacy per-row loop (`MEND_NORM_SEQUENTIAL=1`) - i.e. the single largest
optimisation is now confirmed on the GPU, not just in op counts.

Remaining head-room (not attempted, would change numerics/shape):
`logits_to_keep` for the lm_head (~10 supervised tokens out of 2 964 while the
head spans 152 128 vocab), TF32 for the fp32 transform, vision-tower feature
caching. The 3.0 s/step of pure forward/backward GEMM work is the algorithm.

## 3. Validation event cost (run F, `val_steps=100`, `val_batch_size=1`)

| phase | measured |
|-------|----------|
| pre-validation checkpoint write (3.87 GiB, fsync + rename) | 36.6 s |
| 100 validation samples | 295.4 s (2.954 s/sample) |
| best-model write (3.87 GiB) | 31.0 s |
| **total per validation event** | **~363 s (6.0 min)** |

Cross-check with the last completed 30 000-step run (`MEND_LLAVAOV_VQA_NORMTRUE_META_TRAIN`,
09/03): validations there took 489-497 s at its 7.4 s/step-class speed, i.e.
the same ~4.9 val-samples/s ratio.

Note: `BaseTrainer.run` calls the mid-run validation with
`validate(steps=self.config.val_steps)` (100) but the **final** evaluation with
`steps=None` -> `len(val_set)`; with the frozen `val_size=1000` that is a
1 000-sample final evaluation (~50 min), included in the ETA below.

## 4. ETA for the configured 30 000-step run

| | train | 5 validations | final eval | startup | total | vs historic |
|---|-------|---------------|-----------|---------|-------|-------------|
| pre-optimisation (7.424 s/step) | 61.87 h | 0.64 h | 1.11 h | 0.02 h | **63.64 h** | - |
| frozen config, `workers=0` (5.487 s/step) | 45.73 h | 0.50 h | 0.82 h | 0.02 h | **47.06 h** | **-16.58 h (-26.1%)** |
| with `PMRL_MEND_NUM_WORKERS=4` (5.200 s/step) | 43.33 h | 0.50 h | 0.82 h | 0.02 h | **44.67 h** | -18.97 h (-29.8%) |

If the final evaluation turns out to be the 100-sample mid-run protocol instead
of the full 1 000-sample `val_set`, subtract ~0.73 h (46.33 h at `workers=0`).

Checkpoint schedule at 5.487 s/step (plus 6 min per validation), useful for
aligning a planned shutdown with a recoverable state:

```
step  5000 ~  7.6 h      step 20000 ~ 30.5 h
step 10000 ~ 15.2 h      step 25000 ~ 38.1 h
step 15000 ~ 22.9 h      step 30000 ~ 45.7 h
```

## 5. Numerical side: is the optimised path still the same experiment?

`tests/test_mend_perf_equivalence.py` pins the CPU-side equivalence. On the GPU
the picture is:

| comparison | loss/edit rel. dev | acc metrics | `diag/outer_nonzero` |
|------------|--------------------|-------------|----------------------|
| step 1: block vs legacy Welford (G vs H) | 1.7e-04 | edit/inner/loc acc identical | identical (13) |
| step 10: A vs D | 3.1e-04 | identical | identical |
| step 20: A vs D | 1.2e-03 | loc/acc 0.1406 vs 0.1385 | identical |
| step 10: **identical** config, RA vs RB | 5.2e-04 | identical | identical |
| step 20: **identical** config, RA vs RB | 7.1e-04 | loc/acc 0.1406 vs 0.1451 | identical |
| step 20: RA vs RB, `grad_train` | - | - | 2.5e-03 |

Two runs of the *identical* binary, config and seed already differ by ~5e-04
at step 10 and ~2.5e-03 (gradient norm) by step 20: this pipeline is not
bit-reproducible (bf16 kernels + non-deterministic reductions, no
`use_deterministic_algorithms`). The block-vs-legacy deviation is the same
order of magnitude or smaller, and the connectivity invariants
(`diag/outer_nonzero_train`, `diag/*_requires_grad_train`) are unchanged, so
there is **no evidence of a semantic change** - but the report's earlier
"5 significant digits" acceptance criterion is not achievable on this box
beyond the first step, for any code variant.

Consequence for the experiment: the optimised run will *not* reproduce the
09/10 trajectory metric-by-metric beyond the first few dozen steps. That is a
property of the pipeline, not of the optimisation. Any future equivalence claim
must be made on the step 1-3 window or the unit tests, not on 100-step
trajectories.

Claims that are **verified**: step time (-1.70 s/step same-day A/B), norm-stats
cost (5.5 ms vs 1684.5 ms), data-pipeline overlap (`workers=4`: 5.200 vs
5.487 s/step, residual 0.483 vs 0.655 s), validation/checkpoint costs, exit
codes 0 for all seven runs, no traceback in any log.
Claims that are **not** verified: that the 30 000-step optimised run reaches the
same validation curve as the 09/10 run (cannot be verified before running it),
and the `val_size=1000` final evaluation size (code-read, not measured).

## 6. Disk blocker (resolved 2026-09-12)

`/root` was at **98 % (7.3 GiB free)**; each MEND checkpoint is 3.87 GiB and each
validation event leaves `<name>`, `<name>.bk` and `<name>.prevalidation`, so the
run needs ~11.6 GiB free plus ~0.3 GiB of transient write space, and it writes
at every 5 000-step validation. Reclaimable engineering artifacts
(no results, no metrics):

```
results/MEND_LLAVAOV_LAP_PMRL_SMOKE              2.6G
results/MEND_LLAVAOV_LAP_PMRL_SMOKE2             2.6G
results/MEND_LLAVAOV_LAP_PMRL_DIAG               2.6G
results/MEND_LLAVAOV_LAP_PMRL_GRAPHFIX_SMOKE_FRESH 7.8G
results/MEND_LLAVAOV_LAP_PMRL_IC_META_TRAIN_RETRY3 3.9G
results/MEND_LLAVAOV_VQA_VALIDATION_N100         3.9G
results/models (stray, mis-set results_dir)      7.8G
results/MEND_LLAVAOV_IC_PERF_TIMING_100 (this work) 7.8G
                                       total   ~39 GiB
```

The results to keep (baseline + enhanced, per the project convention) are the
`MEND_LLAVAOV_*_META_TRAIN*`, `MEND_QWEN2VL_*_META_TRAIN*` and `*_N100`
directories; none of those were touched.

---

## 7. Full run: attempt 1 (crashed), root cause, fix, attempt 2 (live)

Cleanup done first (`run_logs/cleanup_20260912.json`): 7 engineering
directories removed, 30.98 GiB freed, `/root` 12 -> 43 GiB free. Kept:
all `MEND_*_META_TRAIN*` / `*_N100` result directories.

**Attempt 1** (`run_logs/mend_llavaov_ic_full_asam_20260912.log`,
launcher `run_logs/run_full_asam_20260912.sh`, started 14:03:54 with
`PMRL_MEND_NUM_WORKERS=4`) died at 15:32:30 with `EXIT_CODE=1` after reaching
step 1000:

```
OSError: Caught OSError in DataLoader worker process 0.
  PIL/PngImagePlugin.py -> struct.error: unpack_from requires a buffer of at
  least 4 bytes ... (actual buffer size is 0)  ->  OSError: image file is truncated
```

**Not corrupt data**: `tests/scan_caption_image_integrity.py` fully decoded all
4 946 distinct image files referenced by the 1000 train + 1000 eval records
(byte sizes vs directory medians) - **0 unusable**
(`run_logs/image_integrity_scan_20260912.json`).

**Root cause** (reproduced without a model by
`tests/diagnose_caption_dataloader.py --workers 4 --size 1000 --epochs 3`, which
fails at batch 1001 pre-fix): `CaptionDataset.__init__` stored *lazily-opened*
`PIL.Image` objects (one open file descriptor per image). The dataset is built in
the parent and then forked into the 4 workers, so all workers inherited the same
open file descriptions - one shared file offset per image - while each kept its
own empty decode cache. A record decoded by worker A in epoch 1 was re-read by
worker B in epoch 2 starting at the already-consumed shared offset -> EOF ->
"image file is truncated". With `workers=0` the single process always re-reads
its own cached pixels, which is why the historic runs and every sub-epoch smoke
(including the 40-step `workers=4` timing run, which never crossed an epoch
boundary) were clean.

**Fix**: `easyeditor/dataset/coco_caption.py` gains `_materialise()`, which
decodes each vision input at construction and detaches it from the file
(`load()` + `copy()` + `close()`), so workers never touch a file handle. The
decode also moves out of the per-step collate. Pixel equality verified for both
a JPEG and a PNG rephrase image (identical arrays, `fp` gone).

**Verification of the fix** (`run_logs/verify_dataloader_fix_20260912.txt`,
`..._control_20260912.txt`):

| check | result |
|-------|--------|
| `diagnose_caption_dataloader.py --workers 4 --size 1000 --epochs 3` | 3000 batches across 3 epoch boundaries, no error (pre-fix: died at 1001) |
| same, `--workers 0` control | 600 batches, no error |
| `pytest tests/ -q` | **80 passed** (78 + 2 new: `tests/test_caption_dataloader_multiprocess.py`) |
| dataset init cost | 1.7 s -> 26.6 s per 1000 records (one-time, ~55 s per run) |

**Attempt 2** (live): `run_logs/run_full_asam.sh` ->
`run_logs/mend_llavaov_ic_full_asam_20260912b.log`, started 15:51:09 with
`PMRL_MEND_NUM_WORKERS=4`, same frozen config (30000 / val_interval 5000 /
val_steps 100 / final_eval / checkpoint_before_validation), 43 GiB free,
10-minute disk watchdog. ETA **44.69 h -> 2026-09-14 12:32**, checkpoint
milestones in
`results/MEND_LLAVAOV_LAP_PMRL_IC_GRAPHFIX_FINAL2/run_manifest.json`
(attempt 1's manifest kept as `run_manifest_attempt1_failed.json`).
