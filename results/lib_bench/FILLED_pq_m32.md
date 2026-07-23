# Library hipVS vs cuVS — IVF_PQ m=32 (2026-07-23)

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

FLAT on AMD still needs re-run with `--p99-sample 0`.
