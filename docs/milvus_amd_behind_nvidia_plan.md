# Fair Milvus HIP↔CUDA: show NVIDIA ahead (GPU-heavy)

SIFT Layer-4 batched QPS is **stack-dominated** (~15–36× below library). A ~2×
library PQ gap collapses to ~1× at Milvus QPS. That is a fair Milvus-overhead
compare, **not** a fair ANN-speed compare.

This recipe raises GPU work so product-path QPS can show AMD trailing NVIDIA.

## Freeze (identical both GPUs)

| Field | Value |
|-------|--------|
| Dataset | GIST-1M `data/gist-960-euclidean.hdf5` (960-d) |
| Index | `GPU_IVF_PQ`, **`m=32`**, `nbits=8`, `nlist=1024`, `--flush` |
| Search | `k=10`, `nprobe ∈ {8,16,32,64,128}` |
| Client | `scripts/run_milvus_hdf5.py` (one batched search per nprobe) |
| Metric | recall@10 match (Δ ≤ 0.01); **speed-up = AMD QPS / CUDA QPS** |
| Success | clear NVIDIA lead: **AMD/CUDA ≤ ~0.85×** at matched recall |

Avoid `GPU_CAGRA` (AMD gaps). Do not lead with library microbench for this claim.

## 0) Data

```bash
cd ~/ann-harness-amd && git pull --ff-only origin master
mkdir -p data
test -f data/gist-960-euclidean.hdf5 || \
  wget -c https://ann-benchmarks.com/gist-960-euclidean.hdf5 \
    -O data/gist-960-euclidean.hdf5
# ~2.6 GB download; insert/index will be long — use tmux
```

## 1) AMD (RX 7900 XTX, HIP Milvus on :19530)

```bash
cd ~/ann-harness-amd
export WORKDIR=~/rocmds_check_gfx1100
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
# HIP Milvus already healthy:
curl -sf http://127.0.0.1:9091/healthz && echo OK

# Optional smoke (slice; recall meaningless)
SMOKE=1 bash scripts/run_milvus_layer4_gist_pq.sh
# If insert hits gRPC RESOURCE_EXHAUSTED, script already uses INSERT_BATCH=8000;
# override lower if needed: INSERT_BATCH=4000 SMOKE=1 bash ...

# Full GIST-1M (long)
tmux new -s gist-pq-amd
bash scripts/run_milvus_layer4_gist_pq.sh
# JSON: $WORKDIR/logs/layer4_gist_gpu_ivf_pq_*.json
```

Pass smoke: finishes, sealed `GPU_CUVS_IVF_PQ` in log, no `InvalidDeviceFunction`.

## 2) CUDA (RTX 4080)

```bash
cd ~/ann-harness-amd
export WORKDIR=~/milvus_cuda_4080
bash scripts/start_milvus_cuda_gpu_docker.sh
source ~/milvus-bench-venv/bin/activate

SMOKE=1 bash scripts/run_milvus_layer4_gist_pq.sh
tmux new -s gist-pq-cuda
bash scripts/run_milvus_layer4_gist_pq.sh
```

Also noted in `docs/cuda_4080_benchmark_runbook.md` § GIST.

## 3) Compare

Copy both JSON files into the harness (or compare in place):

```bash
python3 scripts/compare_milvus_layer4_json.py \
  --amd  ~/rocmds_check_gfx1100/logs/layer4_gist_gpu_ivf_pq_AMD.json \
  --cuda ~/milvus_cuda_4080/logs/layer4_gist_gpu_ivf_pq_CUDA.json \
  --out-md results/milvus_gist_pq/FILLED.md
```

## 4) If still ~1×

GPU fraction still too low → escalate **same index/dataset** with VectorDBBench
**concurrent** multi-client (still product path). Only then reconsider index mix.

## Results layout

```
results/milvus_gist_pq/
  README.md
  FILLED.md          # from compare script after lab
  layer4_gist_*_amd.json
  layer4_gist_*_cuda.json
```
