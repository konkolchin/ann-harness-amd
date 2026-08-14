# Milvus GIST GPU_IVF_FLAT — AMD vs CUDA (2026-08-14)

Fair product-path compare: GIST-1M (960-d), sealed `GPU_IVF_FLAT`,
nlist=1024, k=10, batched search, nprobe ∈ {8,16,32,64,128}.
Speed-up = AMD QPS / CUDA QPS.

| nprobe | AMD QPS (7900) | CUDA QPS (4080) | R@10 AMD | R@10 CUDA | AMD/CUDA |
|--------|----------------|-----------------|----------|-----------|----------|
| 8 | 1643 | 1339 | 0.648 | 0.640 | **1.23×** |
| 16 | 4925 | 4648 | 0.780 | 0.775 | **1.06×** |
| 32 | 4158 | 4103 | 0.890 | 0.883 | **1.01×** |
| 64 | 2554 | 3774 | 0.957 | 0.954 | **0.68×** |
| 128 | 2256 | 1961 | 0.989 | 0.989 | **1.15×** |

**Verdict:** recall ladders match (Δ ≤ ~0.01). QPS is mostly peer / AMD-ahead
except **nprobe=64** (AMD 0.68×). First point (`nprobe=8`) is cold on both
GPUs (QPS rises at 16) — optional re-run with `SEARCH_WARMUP=5 SEARCH_RUNS=10`.

Sources:
- AMD: `layer4_gist_gpu_ivf_flat_20260814_221631.json`
  (`~/rocmds_check_gfx1100/logs/`)
- CUDA: `layer4_gist_gpu_ivf_flat_20260814_223436.json`
  (`~/milvus_cuda_4080/logs/`)

```bash
python3 scripts/compare_milvus_layer4_json.py \
  --amd  results/milvus_gist_flat/layer4_gist_gpu_ivf_flat_20260814_221631.json \
  --cuda results/milvus_gist_flat/layer4_gist_gpu_ivf_flat_20260814_223436.json \
  --out-md results/milvus_gist_flat/FILLED.md
```
