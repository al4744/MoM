# Throughput sweep — saturation at p95 post-tool TTFT < 500 ms

## Saturation rates

| config | sustained λ at SLO (RPS) |
|---|---|
| baseline_filler | — |
| prefix_cache_only_filler | **1** |
| retention_filler | — |

## Raw curve (post-tool TTFT)

| config | rate (RPS) | p95 (ms) | p99 (ms) | mean (ms) | n |
|---|---|---|---|---|---|
| baseline_filler | 1 | 819.6 | 1041.9 | 246.6 | 450 |
| baseline_filler | 2 | 1474.8 | 2084.1 | 544.7 | 450 |
| baseline_filler | 4 | 11386.7 | 12294.2 | 4696.7 | 450 |
| baseline_filler | 8 | 10845.4 | 11840.9 | 7380.1 | 450 |
| baseline_filler | 16 | 15317.5 | 15715.9 | 10337.6 | 450 |
| prefix_cache_only_filler | 1 | 495.8 | 690.7 | 175.9 | 450 |
| prefix_cache_only_filler | 2 | 2793.4 | 4189.4 | 1042.9 | 450 |
| prefix_cache_only_filler | 4 | 11243.3 | 11806.8 | 4853.9 | 450 |
| prefix_cache_only_filler | 8 | 10428.6 | 11239.3 | 7146.8 | 450 |
| prefix_cache_only_filler | 16 | 28444.3 | 29784.5 | 18688.0 | 450 |
| retention_filler | 1 | 668.1 | 892.7 | 231.3 | 450 |
| retention_filler | 2 | 5467.1 | 6338.9 | 1697.6 | 450 |
| retention_filler | 4 | 11061.9 | 11753.8 | 4744.4 | 450 |
| retention_filler | 8 | 10352.1 | 11582.8 | 7243.4 | 450 |
| retention_filler | 16 | 29095.8 | 30602.8 | 19391.5 | 450 |
