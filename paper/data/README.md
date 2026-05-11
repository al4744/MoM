# Paper data artifacts

| Directory | Tables in paper |
|---|---|
| `isolation/` | Table 1 (Daksh microbench events log) |
| `throughput-abundant/` | Table 4 (gpu_util=0.85 sweep) |
| `throughput-constrained/` | Table 5 (gpu_util=0.30 sweep) |
| `accuracy-fp16/` | Table 3 (FP16 baseline vs retention) |

`saturation_curve.json` has `curve.<config>[i].p95_ms` keyed by arrival rate (`rates_swept`).
`events_*.jsonl` is newline-delimited PinManager events (pin / reuse / evict / expire).
