# Library hipVS vs cuVS — IVF_PQ m=32

## Baseline (2026-07-23) — stock launch heuristic

| nprobe | hipVS QPS | cuVS QPS | R@10 both | speed-up (hip/cu) |
|--------|-----------|----------|-----------|-------------------|
| 1 | 1,593,707 | 4,326,105 | 0.35 | 0.37× |
| 4 | 1,268,942 | 2,297,667 | 0.59 | 0.55× |
| 8 | 710,989 | 1,411,722 | 0.67 | 0.50× |
| 16 | 393,742 | 806,267 | 0.71 | 0.49× |
| 32 | 217,275 | 453,316 | 0.72 | 0.48× |

Sources:
- `lib_hipvs_ivf_pq_m32_20260723_211759.json`
- `lib_cuvs_ivf_pq_m32_20260723_201837.json`

## After AMD default `blockDim=512` (2026-08-15)

Hardcoded in lab hipVS / `konkolchin/hipVS` (`aae4bbe` + harness `0006`).
Stderr: `[ann-harness] … block_threads=512 …`. Spot‑1 compute_score opt **reverted** (no win).

| nprobe | hipVS QPS (bt512) | cuVS QPS (unchanged) | R@10 hip | speed-up (hip/cu) |
|--------|-------------------:|---------------------:|---------:|------------------:|
| 1 | 1,733,219 | 4,326,105 | 0.35 | **0.40×** |
| 4 | 1,177,594 | 2,297,667 | 0.59 | **0.51×** |
| 8 | 862,599 | 1,411,722 | 0.67 | **0.61×** |
| 16 | 497,091 | 806,267 | 0.71 | **0.62×** |
| 32 | 269,845 | 453,316 | 0.73 | **0.60×** |

Source: `lib_hipvs_ivf_pq_m32_20260815_000637.json`

**vs July hipVS @ nprobe=32:** 217k → **270k** (~**+24%**).  
**vs cuVS:** mid-grid gap improves from ~0.48–0.50× to ~**0.60–0.62×** (still not closed).
