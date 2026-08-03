# CAGRA lib / Catch2 status (gfx1100)

## Phase A triage (2026-08-03, amd-rx7900xtx, ROCm 7.0.2)

### Catch2 `Test All GPU Index` — 572/576 (4 failed)

| Line | Index | Got | Need |
|------|-------|-----|------|
| 205 | **GPU_CUVS_CAGRA** | **0.0** | > 0.7 |
| 328 | **GPU_CUVS_CAGRA** | **0.0** | ≥ 0.8 |
| 242 | GPU_CUVS_IVF_PQ | 0.835 | ≥ 0.85 (near-miss) |
| 354 | GPU_CUVS_IVF_PQ | 0.647 | > 0.65 (near-miss) |

### hipVS Python (bypass Knowhere)

| Recipe | Result |
|--------|--------|
| default / `ivf_pq` | **THROW** `graph_core` invalid/duplicated neighbors |
| `build_algo=nn_descent` (hipVS lowercase) | **BUILD OK** (~0.2s on 10k×128) |
| search itopk=64, k=10, N=10k/200 | **recall@10 = 0.0695**, QPS ~1.95e5 |

JSON: `$WORKDIR/logs/lib_hipvs_cagra_20260803_234622.json`  
`verify IndexParams.build_algo=2` (enum for nn_descent).

### Ownership

**hipVS / ROCm-DS on consumer gfx1100**

- IVF-PQ graph build path: hard fail  
- NN-Descent path: runs but **quality unusable** (~7% recall@10 at this smoke size)  
- Knowhere Catch2 recall 0.0 likely uses default IVF-PQ (or broken graph) — do not treat as Knowhere-only until lib recall is healthy

**Phase B blocked** until lib recall is in a useful band (target ≥ ~0.7 at some itopk / degree).

### Next lab tries

```bash
source ~/hipvs-bench-venv/bin/activate
export WORKDIR=~/rocmds_check_gfx1100
cd ~/ann-harness-amd && git pull --ff-only

# A) Larger graph / itopk (quality)
MAX_TRAIN_ROWS=10000 MAX_QUERY_ROWS=200 \
  GRAPH_DEGREE=64 INTERMEDIATE_GRAPH_DEGREE=128 \
  ITOPK_SIZES=64,128,256,512 \
  GRAPH_BUILD_ALGO=nn_descent \
  bash scripts/run_hipvs_cagra_bench.sh

# B) Full SIFT-1M (same degrees) — slower build
GRAPH_BUILD_ALGO=nn_descent ITOPK_SIZES=128,256,512 \
  GRAPH_DEGREE=64 INTERMEDIATE_GRAPH_DEGREE=128 \
  bash scripts/run_hipvs_cagra_bench.sh

# C) CUDA peer (same recipe) on 4080
WORKDIR=~/milvus_cuda_4080 GRAPH_BUILD_ALGO=nn_descent \
  MAX_TRAIN_ROWS=10000 MAX_QUERY_ROWS=200 ITOPK_SIZES=64 \
  bash scripts/run_cuvs_cagra_bench.sh
# then compare_cuvs_lib_json.py
```

Also try `iterative_cagra_search` if nn_descent recall stays ~0.07.
