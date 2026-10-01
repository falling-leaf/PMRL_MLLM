# BLIP-2 UniKE IC: frozen baseline vs ASAM (N=100)

Scope: MMEdit E-IC first 100 effective cases. Frozen comparator is WISE BLIP-2 baseline with extra/LAP/PMRL disabled. Candidate is isolated online-simplified UniKE-BLIP2 ASAM: added key/value pairs in OPT FFNs 20–23 only; frozen backbone/retrieval memory; 10 added pairs/layer, 20 steps, Adam lr=1e-4 eps=1e-4, ASAM epsilon=0.1, ASAM weight=1.0.

Integrity: `{"exit_code_0": true, "sample_count_100": true, "per_case_count_100": true, "case_ids_continuous_0_99": true, "one_final_marker": true, "no_traceback": true, "expected_asam_markers": true, "no_export_markers": true}`. First N=100 attempt was interrupted at 899/2000 markers by foreground tool timeout and discarded. Rerun is complete.

| Method | Acc | Gen-T | Gen-M | Gen avg | Loc-T | Loc-M | Loc avg | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen WISE baseline | 100.00 | 95.91 | 86.51 | 91.21 | 100.00 | 10.23 | 55.12 | 321.00 |
| UniKE-BLIP2 ASAM | 96.37 | 93.21 | 88.89 | 91.05 | 89.26 | 14.63 | 51.95 | 1227.33 |

Paired Gen-M (ASAM - baseline): +2.38 pp; normal 95% CI [-0.86, +5.63] pp; bootstrap 95% CI [-0.91, +5.51] pp (10,000, seed 42); win/tie/loss 46/32/22.

Decision: ASAM is retained but not selected, because the paired bootstrap lower bound is not strictly positive. WISE remains selected. ASAM point Gen-M is +2.38 pp but Acc -3.63 pp, Gen-T -2.71 pp, Loc-T -10.74 pp, and CI crosses zero.

Not paper-equivalent: official UniKE assets/release are MiniGPT-4-only; no BLIP-2-compatible retrieved states, latent-IKE, semantic encoder are local.

Artifacts: baseline `/root/PMRL_MLLM/results/WISE_IC_BLIP2_BASELINE_UNIKE_FROZEN_N100/result.json`; ASAM `/root/PMRL_MLLM/results/UNIKE_BLIP2_IC_ASAM_N100_RERUN/result.json`; registry `/root/PMRL_MLLM/results/UNIKE_BLIP2_IC_registry.json`.
