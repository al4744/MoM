# Throughput sweep — saturation at p95 post-tool TTFT < 500 ms

## Saturation rates

| config | sustained λ at SLO (RPS) |
|---|---|
| baseline_filler | — |
| prefix_cache_only_filler | — |
| retention_filler | — |

## Raw curve (post-tool TTFT)

| config | rate (RPS) | p95 (ms) | p99 (ms) | mean (ms) | n |
|---|---|---|---|---|---|
| baseline_filler | 1 | 5344.9 | 6717.6 | 2056.2 | 450 |
| baseline_filler | 2 | 22320.8 | 23124.9 | 13279.9 | 450 |
| baseline_filler | 4 | 30200.5 | 30695.8 | 20885.9 | 450 |
| baseline_filler | 8 | 34766.1 | 35185.5 | 26772.1 | 450 |
| baseline_filler | 16 | 34953.4 | 35343.5 | 28052.5 | 450 |
| prefix_cache_only_filler | 1 | 9883.9 | 12173.5 | 3690.1 | 450 |
| prefix_cache_only_filler | 2 | 27621.4 | 28652.9 | 14553.9 | 450 |
| prefix_cache_only_filler | 4 | 40423.0 | 42555.1 | 23414.0 | 450 |
| prefix_cache_only_filler | 8 | 35448.3 | 36183.8 | 27157.7 | 450 |
| prefix_cache_only_filler | 16 | 36151.4 | 37546.8 | 30259.9 | 450 |
| retention_filler | 1 | 7120.0 | 9721.6 | 2459.3 | 450 |
| retention_filler | 2 | 26412.5 | 28033.0 | 14463.6 | 450 |
| retention_filler | 4 | 37534.4 | 38985.2 | 22765.6 | 450 |
| retention_filler | 8 | 34517.6 | 35611.2 | 26663.8 | 450 |
| retention_filler | 16 | 35339.4 | 36527.4 | 29353.8 | 450 |
