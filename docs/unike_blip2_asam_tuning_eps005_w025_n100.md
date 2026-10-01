# UniKE-BLIP2 ASAM tuning: eps=0.05, weight=0.25 (N=100)

N=10 serial tuning gate selected eps=.05/w=.25 for N=100 confirmation. N=10 was only a gate.

| Method | Acc | Gen-T | Gen-M | Gen avg | Loc-T | Loc-M | Loc avg | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen WISE baseline | 100.00 | 95.91 | 86.51 | 91.21 | 100.00 | 10.23 | 55.12 | 321.00 |
| UniKE ASAM eps=.05,w=.25 | 96.27 | 93.53 | 88.68 | 91.10 | 90.10 | 15.48 | 52.79 | 1244.43 |

Gen-M paired delta: +2.17 pp; normal CI [-0.86,+5.20]; bootstrap CI [-0.87,+5.12] pp; win/tie/loss 41/36/23.

Integrity: `{"exit_code_0": true, "sample_count_100": true, "per_case_count_100": true, "ids_0_99": true, "one_final": true, "no_traceback": true, "asam_markers_2000": true, "no_export": true}`.

Decision: ASAM is retained but not selected because bootstrap lower CI is not >0. Frozen WISE remains selected. Gen-M point estimate is +2.17 pp, but Acc −3.73 pp, Gen-T −2.39 pp, Loc-T −9.90 pp; CI crosses zero.

Artifacts: `/root/PMRL_MLLM/results/UNIKE_BLIP2_IC_ASAM_TUNE_EPS005_W025_N100/result.json`, `/root/PMRL_MLLM/run_logs/unike_blip2_ic_asam_tune_eps005_w025_n100/run.log`, `/root/PMRL_MLLM/results/UNIKE_BLIP2_IC_registry.json`.
