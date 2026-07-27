# Milvus GIST-1M GPU_IVF_PQ (fair HIP↔CUDA)

GPU-heavy product-path compare — see [milvus_amd_behind_nvidia_plan.md](../../docs/milvus_amd_behind_nvidia_plan.md).

## Status

| Side | Status |
|------|--------|
| AMD RX 7900 XTX (HIP Milvus) | **PENDING lab run** |
| NVIDIA RTX 4080 (CUDA Milvus) | **PENDING lab run** |
| Compare (`FILLED.md`) | **PENDING** |

Lab hosts are not reachable from the Cursor agent. Run on the boxes:

```bash
# AMD
cd ~/ann-harness-amd && git pull
export WORKDIR=~/rocmds_check_gfx1100
bash scripts/run_milvus_layer4_gist_pq.sh

# CUDA
export WORKDIR=~/milvus_cuda_4080
bash scripts/start_milvus_cuda_gpu_docker.sh
bash scripts/run_milvus_layer4_gist_pq.sh

# Compare → copy JSON here then:
python3 scripts/compare_milvus_layer4_json.py \
  --amd  results/milvus_gist_pq/layer4_gist_*_amd.json \
  --cuda results/milvus_gist_pq/layer4_gist_*_cuda.json \
  --out-md results/milvus_gist_pq/FILLED.md
```

## Expected filenames

- `layer4_gist_gpu_ivf_pq_<TS>_amd.json` (or copy from `$WORKDIR/logs/`)
- `layer4_gist_gpu_ivf_pq_<TS>_cuda.json`
- `FILLED.md` — AMD/CUDA QPS + recall + speed-up
