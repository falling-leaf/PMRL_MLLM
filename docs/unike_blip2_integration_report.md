# UniKE → PMRL_MLLM integration status (2026-08-19)

## Scope and source inspection
- Paper inspected: `/root/UniKE.pdf` (19 pages; extracted text: `/root/unike_reference/UniKE.txt`).
- Official code cloned at `/root/UniKE`, revision `dcf267e` (`Update readme.md`).
- Target task/model: BLIP-2 OPT-2.7B, MMEdit E-IC, first 100 effective records.

## What the paper and released implementation actually do
UniKE combines: (1) intrinsic paired key/value additions in the last four FFNs; (2) external latent-IKE feature shifting using top-40 retrieved hidden states in those layers; and (3) a dynamic cosine semantic gate based on a pretrained semantic encoder. Appendix D.3 specifies ten added key/value pairs in the final four layers and top-40 retrieved states.

The released implementation is MiniGPT-4/Vicuna-only. It requires three artifact classes from Hugging Face `Parva1012/UniKE_ckpt`: per-case `ike_<id>.pth` states, `l-ike.pth`, and `semantic_encoder.pt`. Local inventory found none. These MiniGPT-4 tensors/checkpoints cannot truthfully be used with BLIP-2 OPT.

## PMRL integration
Added isolated key `UniKE-BLIP2`; no WISE/PMRL/LAP config or code path was changed:
- `/root/PMRL_MLLM/easyeditor/models/unike_blip2/unike_blip2_hparams.py`
- `/root/PMRL_MLLM/easyeditor/models/unike_blip2/unike_blip2_main.py`
- `/root/PMRL_MLLM/hparams/UniKE/blip2_ic_unike_online_simplified.yaml`
- `/root/PMRL_MLLM/run_unike_blip2_ic.py`
- `/root/PMRL_MLLM/tests/test_unike_blip2.py`

The fallback is explicitly `online_simplified`, not paper-equivalent: it uses each edit's BLIP-2 activations as frozen per-case retrieval memory, preserves four-layer paired FFN expansion, top-40 retrieval, cosine gating, norm-preserving shifts, and 20-step intrinsic optimization. It does not use official latent-IKE/semantic encoder/retrieved memory artifacts.

## Verification
- Focused tests + existing WISE mode tests: `16 passed`.
- Python compilation: passed.
- Real UniKE-BLIP2 simplified N=1 smoke: exit 0. Artifact: `/root/PMRL_MLLM/results/UNIKE_BLIP2_IC_SMOKE1/result.json`.

## Frozen BLIP-2 IC baseline (completed)
Method: existing WISE baseline, `using_extra=false`, `using_lap=false`, `using_pmrl=false`; frozen comparator before UniKE.

| N | Acc | Gen-T | Gen-M | Loc-T | Loc-M | Gen avg | Loc avg | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 100.00 | 95.91 | 86.51 | 100.00 | 10.23 | 91.21 | 55.12 | 321.00 |

Checks: `{"exit_code_0": true, "sample_count_100": true, "per_case_count_100": true, "case_ids_continuous_0_99": true, "one_final_marker": true, "no_traceback": true, "no_pmrl_markers": true}`.

Artifacts:
- Result: `/root/PMRL_MLLM/results/WISE_IC_BLIP2_BASELINE_UNIKE_FROZEN_N100/result.json`
- Metrics: `/root/PMRL_MLLM/results/WISE_IC_BLIP2_BASELINE_UNIKE_FROZEN_N100/metrics.txt`
- Log: `/root/PMRL_MLLM/run_logs/unike_blip2_baseline_n100/run.log`
- Status: `/root/PMRL_MLLM/run_logs/unike_blip2_baseline_n100/status.txt`

## Original-versus-simplified comparison status
Not runnable yet: official UniKE supplies MiniGPT-4 artifacts, and no BLIP-2-compatible official state/checkpoint is local. Therefore an N=100 simplified result would not be a valid original-vs-simplified comparison. It requires BLIP-2-compatible per-case `ike_0.pth` ... `ike_99.pth`, `l-ike.pth`, and `semantic_encoder.pt`, supplied manually under the download policy. The runner reports `paper_equivalent=false` until then.
