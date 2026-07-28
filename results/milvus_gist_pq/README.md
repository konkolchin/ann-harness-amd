# Milvus GIST-1M GPU_IVF_PQ (fair HIP↔CUDA)

GPU-heavy product-path compare — see [milvus_amd_behind_nvidia_plan.md](../../docs/milvus_amd_behind_nvidia_plan.md).

## Status

| Side | Status |
|------|--------|
| AMD RX 7900 XTX (HIP Milvus) | **DONE** `layer4_gist_gpu_ivf_pq_20260728_100624.json` |
| NVIDIA RTX 4080 (CUDA Milvus) | **DONE** `layer4_gist_gpu_ivf_pq_20260728_095805.json` |
| Compare | **[FILLED.md](FILLED.md)** — AMD/CUDA ≈ **0.60–0.80×** |

Copy JSON into this folder when convenient:

```bash
# from AMD
scp ~/rocmds_check_gfx1100/logs/layer4_gist_gpu_ivf_pq_20260728_100624.json \
  user@laptop:ann-harness-amd/results/milvus_gist_pq/

# from CUDA
scp ~/milvus_cuda_4080/logs/layer4_gist_gpu_ivf_pq_20260728_095805.json \
  user@laptop:ann-harness-amd/results/milvus_gist_pq/
```
