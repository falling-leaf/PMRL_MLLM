# MEND + LLaVA-OV + ASAM(extra/LAP/PMRL) IC: step-time audit and optimisations

Scope: `PMRL_MLLM`, experiment group "MEND LLaVA ASAM IC", i.e. the meta-training
run `hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-graphfix-final2.yaml`
(`run_mend_llavaov_ic_train.py`, results dir
`results/MEND_LLAVAOV_LAP_PMRL_IC_GRAPHFIX_FINAL2`).

Baseline evidence: `run_logs/mend_llavaov_ic_graphfix_final2.log` reached step
2200 in 15,591 s (step 100 at 17:14:19, step 2200 at 21:34:10) = **7.42 s/step**,
i.e. ~62 h for the configured `max_iters: 30000`. The run was interrupted by
hand (`EXIT_CODE=130`).

Status of this work: every change below is implemented and pinned by CPU
equivalence tests - `pytest tests/ -q` on this box is **78 passed**, 19 of them
new (`tests/test_mend_perf_equivalence.py`, `tests/test_mend_profiling.py`), and
the CPU evidence sections of `tests/perf_evidence_mend_ic.py` (`data`, `ops`)
were re-run here with the §2-§3 numbers reproduced.

The GPU-side wall-clock gain is **now measured** (2026-09-12, RTX 6000D box):
`docs/mend_llavaov_ic_timing_gpu_measurement.md` and
`results/mend_llavaov_ic_perf_timing.json`.  Headline: 7.424 -> 5.487 s/step
(-26 %), `mend.norm_stats` 1684.5 -> 5.5 ms/step (same-day A/B with
`MEND_NORM_SEQUENTIAL=1`), `dataloader_num_workers=4` a further -0.29 s/step,
30 000-step ETA 63.6 h -> 47.1 h (44.7 h with workers).  The measured pipeline
is not bit-reproducible (two identical runs differ by ~5e-4 at step 10), so the
"5 significant digits" criterion in §6.2 below applies to the first step only;
see §5 of the timing document for what is and is not verified.  The earlier
sections of this report remain op-count/timing-derived estimates.

Verification evidence (ad-hoc, no canonical runner in this repo; sessions run
pytest explicitly, one child at a time because of the 2 GiB cgroup cap):
2026-09-12 - `tests/test_mend_profiling.py` 3 passed, `tests/test_mend_perf_equivalence.py` 14 passed, full `tests/` 78 passed.

---

## 1. What one training step actually does

`MultimodalTrainer.edit_step` (`easyeditor/trainer/MultimodalTrainer.py`), batch
size 1, `accumulate_bs: 2`:

| # | call | batch | seq len | images |
|---|------|-------|---------|--------|
| 1 | `self.model(batch["loc"])` (no_grad) | text-only locality | 31 | 0 |
| 2 | `self.model(batch["loc_image"])` (no_grad) | image locality | 2953 | 5 tiles |
| 3 | `self.model.edit(batch["edit_inner"])` | inner edit | 2964 | 5 tiles |
| 4 | `edited_model(batch["edit_outer"])` | rephrase prompt | 2970 | 5 tiles |
| 5 | `edited_model(batch["edit_inner"])` | metric only | 2964 | 5 tiles |
| 6 | `edited_model(batch["edit_outer_image"])` | rephrase image | 3720 | 5 tiles |
| 7 | `edited_model(batch["loc"])` | text locality | 31 | 0 |
| 8 | `edited_model(batch["loc_image"])` | image locality | 2953 | 5 tiles |

`MEND.edit` (call 3) itself runs, inside one step: the edit forward, then
`_compute_hf_lap_pmrl_loss` = LAP probe forward + `num_rephrase`(2) variant
forwards, then `loss.backward()` over the edit graph **and both variant graphs**
(the backward hooks pair each variant's activation with its gradient - that is
the LAP signal), then the gradient transform of both shared transforms
(`GradientTransform`, dims 3584+18944, IDMLP rank 1920) and the weight update
`einsum("bi,bj->ji")`.

Measured by `tests/perf_evidence_mend_ic.py`:

* 11 model forwards/step (8 long: 2.95k-3.72k tokens, 2 text-only at 31 tokens,
  plus the 3 inside `edit()`), of which ~7 run the vision tower;
* 3 full backward passes inside `edit()` + the meta backward over 5 post-edit
  graphs;
* only **10** supervised tokens per batch (`labels != -100`) while every
  forward's lm_head spans the full 152,128-token vocabulary;
* `2*N*L` for one 2,964-token forward at N=7.5e9 = **44.5 TFLOP**.

7.42 s/step for ~11 forwards + ~6-7 forward-equivalents of backward implies
**~0.40 s per forward**, i.e. the step is dominated by real GEMM/backward
compute at a plausible ~40-50% MFU. The removable part is the Python/launch
overhead on top - which is what the changes below attack. There is no
"redundant forward" to delete without changing the algorithm (see §5).

---

## 2. Data pipeline (CPU, measured)

`CaptionDataset.__init__` opens 3 images per record (1.8 s for 4 records) and
keeps PIL objects; all resizing/cropping/normalising happens per step in
`collate_fn` through the *slow* LLaVA-OV image processor
(`use_fast` unset ⇒ slow processor): **0.29-0.46 s per step**, previously fully
serialised with GPU work (`DataLoader(..., )` with no `num_workers`).

Per sub-batch: 2943 / 2943 / 3699 / 0 / 2929 image tokens for
edit_inner / edit_outer / edit_outer_image / loc / loc_image.

---

## 3. Changes delivered

| # | change | mechanism | expected effect | equivalence evidence |
|---|--------|-----------|-----------------|----------------------|
| 1 | Block Welford in `GradientTransform._update_running_stats` | the historic per-row Python loop issued **14.01 tensor ops/row/counter = 480,359 ops/step** (measured with `torch.profiler`); the block form issues ~172 | **1.4-2.4 s/step** at 3-5 µs per CUDA launch (19-33% of the step) | `batch_update_counter` == sequential `update_counter`: exact in float64; in float32 max abs dev 3.4e-07 on the mean, 1.9e-06 on std, `k` bit-identical. `MEND_NORM_SEQUENTIAL=1` restores the OLD loop bit-for-bit |
| 2 | `_FunctionalModel` gets only the edited tensors | `torch.func.functional_call(..., strict=False)` keeps the module's own tensors for absent names, so re-passing ~2k untouched parameters only makes the accessor swap/restore them twice per forward (6 forwards/step) | ~0.05-0.3 s/step (op-count argument, not measured) | partial vs full dict: bit-identical outputs and grads on a LLaVA-shaped toy model |
| 3 | inner-example forward under `no_grad` | its logits feed only `edit_loss_fn` metrics, no `l_*` term; the graph was built and retained for a forward that is never differentiated | 0.05-0.2 s/step + several GB | source guard test + the dataset that *does* reuse these logits (`ComprehendEditDataset`) keeps the grad path |
| 4 | locality softmax/top-k under `no_grad` | `post_base_logits` carries autograd history; the metric softmax/top-k kept its output + a sorted copy for backward | ~0.02-0.05 s/step + ~2-4 GB | indices/acc bit-identical under `no_grad` (test) |
| 5 | `_hf_target_loss` no longer `.contiguous()`s the full logits | boolean indexing works on the slice; the copy was 900 MB/call (4 calls/step) plus a same-size scatter in its backward | ~0.05 s/step + ~3.6 GB | cross-entropy bit-identical (test) |
| 6 | CUDA cache flush only under memory pressure (`gpu_cache_free_threshold_gb`, default 8 GB free) | the unconditional per-step `empty_cache()` returns ~60 GB to the driver and forces `cudaMalloc` again next step | 0.1-0.5 s/step | allocator-only change; keeps a safety valve when free memory < threshold |
| 7 | `grad/*` weight diagnostics behind `mend_log_weight_diagnostics` (default False) | 12 blocking `.item()` syncs/step + weight-sized temporaries, and `RunningStatAverager(exclude=["grad/"])` **dropped these keys anyway**, so they were never logged | ~0.02-0.06 s/step | the cheap `diag/*` connectivity keys stay on; no observable log change |
| 8 | DataLoader workers (`dataloader_num_workers`, default 0 = old behaviour) + `move_to_device` | overlaps the 0.29-0.46 s collate with GPU compute; the collate returns CPU tensors inside (forked) workers and the trainer moves the batch | up to ~0.3 s/step | worker-aware collate is CUDA-free by construction; `move_to_device` walks HF `BatchFeature` (`UserDict`) containers |
| 9 | opt-in profiler `MEND_PROFILE=1` | per-section CUDA-synced timers logged every 20 steps | measurement only | no-op when disabled |

Sum of the measured/estimated items: **~2.0-3.5 s/step**, i.e. 7.4 s -> roughly
4.5-5.5 s/step (-25% to -38%), 62 h -> ~40 h for 30k steps. Item 1 alone is the
single largest win; items 6 and 8 depend on the driver/allocator and on being
enabled.

---

## 4. Where the numbers come from

```
PYTHONPATH=. python tests/perf_evidence_mend_ic.py ops    # op counts + fp32 deviation
PYTHONPATH=. python tests/perf_evidence_mend_ic.py data   # collate + seq lengths
python -m pytest tests/test_mend_perf_equivalence.py -q   # equivalence suite
```

The op-count section is a `torch.profiler` (CPU) count of the two
implementations, scaled by the measured row count (3 views x 2964 tokens =
8892 rows per transform, 2 counters, 2 transforms). The container this was run
in caps the process at 2 GiB, hence the small tensors and the two sections
being separately invocable.

---

## 5. Deliberately NOT changed

* **`torch.stack(variant_losses).mean() * 0.0`** in `_compute_hf_lap_pmrl_loss`
  (the `lar_target_loss_weight` defaults to 0). It is numerically a no-op but it
  adds a graph head that reaches the hooked modules, and the LAP (x, Δ) pairs
  are collected by *backward hooks* that pop from a LIFO stack. Removing the
  head can change which activation is paired with which gradient, so it is not
  a safe "optimisation".
* **The 2 extra variant forwards + their 2 backward passes** (~1/3 of the step):
  that is the LAP mechanism itself (multi-view gradient factors).
* **`logits_to_keep` / reduced lm_head**: only ~10 tokens/sequence are graded,
  so restricting the lm_head to those rows would save ~4% of the step and
  several GB, but it changes GEMM shapes and would need `edit_loss_fn` to
  consume reduced logits - a numerics change I cannot prove bit-identical.
* **Vision-tower feature caching**: 3 distinct images are re-encoded 7 times per
  step; ~1-2% of the step for a plumbing change on all forward paths.
* **`torch.set_float32_matmul_precision('high')`** (TF32 for the fp32 transform
  and einsum, ~1-2%): changes the transform numerics.
* **`masked_log_probs`** reshape/upcast of the full logits (~5 GB retained per
  grad call): the same fix as item 5 but this function is shared by WISE,
  Transformer-Patcher, SERAC etc. and its `inner_sent` branch needs the
  full-shape per-token tensor. Left alone on purpose.
* **Validation protocol** (`val_steps: 100` every 5000 steps, plus a full
  1000-sample final validation) - that is the experiment's protocol, ~1.5% of
  the run.

---

## 6. How to verify on the GPU (acceptance criteria)

1. Smoke, with the breakdown:

   ```
   cd /root/PMRL_MLLM
   MEND_PROFILE=1 MEND_PROFILE_EVERY=20 \
   PMRL_MEND_MAX_ITERS=40 PMRL_MEND_VAL_INTERVAL=1000000 \
   PMRL_MEND_HPARAMS=hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-graphfix-final2.yaml \
   python run_mend_llavaov_ic_train.py 2>&1 | tee run_logs/perf_smoke.log
   ```

   The launcher also accepts `PMRL_MEND_NUM_WORKERS` and
   `PMRL_MEND_GPU_CACHE_FREE_GB` so the frozen experiment yaml never has to be
   touched for an A/B.

   `MEND_PROFILE` prints a per-section table (edit.fwd_edit / edit.loss /
   edit.lap_probe / edit.lap_view / edit.pmrl / edit.backward / edit.einsum /
   mend.norm_stats / step.fwd_* / step.meta_backward ...) whose
   `measured_total_s` should be ~7.4 s/step before the fix and ~4.5-5.5 s/step
   after; `mend.norm_stats` should drop from ~1.4-2.4 s to <0.05 s. First step
   is slower (CUDA/allocator warm-up) - use steps 2+.

2. Numerical equivalence on the GPU (same seed, 100 steps each):

   ```
   MEND_NORM_SEQUENTIAL=1 ...   # historic path
   ...                          # default (block update)
   ```
   Compare the logged `loss/edit`, `edit/acc`, `image_loc/acc`, `grad` between
   the two runs: they must agree to ~5 significant digits (float32 reduction
   order only - the unit tests pin the deviation at 1e-6/1e-7). Checkpoints
   (`.../models/MEND/llava-onevision-qwen2-7b-ov-hf`) can be diffed the same
   way.

3. Integrity gates before treating a run as valid: `MEND_META_TRAIN_START`
   present, `MEND_META_TRAIN_DONE` (or a clean stop), no traceback,
   `EXIT_CODE=0`, `diag/outer_nonzero_train` ~= 1.0 (meta-gradient connected,
   unchanged by these edits), `loc/acc`/`image_loc/acc` non-degenerate, and
   `loss/total_edit` tracking the historic trajectory.

4. Only then is a step-time claim reportable, e.g. "100 steps in X s
   (7.42 s/step -> Y s/step, -Z%)".

---

## 7. Rollback

Everything is additive and default-inert except items 1, 3, 4, 5, 7 whose
defaults are the *faster* paths:

* `MEND_NORM_SEQUENTIAL=1` -> historic Welford loop (bit-exact).
* `dataloader_num_workers: 4` -> opt-in only (default 0).
* `gpu_cache_free_threshold_gb: 0` -> never flush the allocator cache
  (default 8 GB free).
* `mend_log_weight_diagnostics: true` -> restore the old `grad/*` diagnostics.
* `git diff` touches only: `easyeditor/trainer/algs/MEND.py`,
  `easyeditor/trainer/algs/profiling.py` (new),
  `easyeditor/trainer/MultimodalTrainer.py`, `easyeditor/trainer/BaseTrainer.py`,
  `easyeditor/trainer/utils.py`, `easyeditor/dataset/coco_caption.py`,
  `easyeditor/trainer/training_hparams/mend_multimodal_training_hparams.py`,
  `run_mend_llavaov_ic_train.py` (env overrides), plus
  `tests/test_mend_perf_equivalence.py`, `tests/perf_evidence_mend_ic.py` and
  this report.  Note that the working tree already carried uncommitted edits to
  several of these files before this work (e.g. `training=training` in the
  `edit` call, the chunked loss helpers), so a raw `git diff` is a mix of both.

`dataloader_num_workers > 0` is only wired for the HF multimodal collate
(`coco_caption.collate_fn`); the MiniGPT-4/BLIP-2 collates move tensors to CUDA
themselves and must stay at 0 workers.
