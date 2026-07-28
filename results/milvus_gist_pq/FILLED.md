# Milvus GIST GPU_IVF_PQ — AMD vs CUDA (2026-07-28)

Fair product-path compare: GIST-1M (960-d), `GPU_IVF_PQ` m=32, nbits=8,
nlist=1024, k=10, batched search. Speed-up = AMD QPS / CUDA QPS.

| nprobe | AMD QPS (7900) | CUDA QPS (4080) | R@10 AMD | R@10 CUDA | AMD/CUDA |
|--------|----------------|-----------------|----------|-----------|----------|
| 8 | 4867 | 7167 | 0.256 | 0.252 | **0.68×** |
| 16 | 2945 | 4102 | 0.264 | 0.262 | **0.72×** |
| 32 | 2228 | 2774 | 0.267 | 0.267 | **0.80×** |
| 64 | 1145 | 1908 | 0.268 | 0.268 | **0.60×** |
| 128 | 700 | 959 | 0.268 | 0.268 | **0.73×** |

**Verdict:** recall matches (Δ ≤ 0.004). NVIDIA ahead on Milvus QPS
(**AMD/CUDA ≈ 0.60–0.80×**, all ≤ 0.85× success target).

Sources:
- AMD: `layer4_gist_gpu_ivf_pq_20260728_100624.json`
  (`~/rocmds_check_gfx1100/logs/`)
- CUDA: `layer4_gist_gpu_ivf_pq_20260728_095805.json`
  (`~/milvus_cuda_4080/logs/`)

Note: AMD single-query p99 (~18–40 ms) ≫ CUDA p99 (~1.7–2.6 ms); primary
metric for this recipe is batched QPS.

```bash
python3 scripts/compare_milvus_layer4_json.py \
  --amd  results/milvus_gist_pq/layer4_gist_gpu_ivf_pq_20260728_100624.json \
  --cuda results/milvus_gist_pq/layer4_gist_gpu_ivf_pq_20260728_095805.json \
  --out-md results/milvus_gist_pq/FILLED.md
```
