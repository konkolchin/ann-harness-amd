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

`cagra.build` on SIFT 10k×128 **throws** during graph optimize:

```text
RAFT failure ... graph_core.cuh ... Could not generate an intermediate CAGRA graph
because the initial kNN graph contains too many invalid or duplicated neighbor nodes.
... overflows occur during the norm computation ...
```

Log showed IVF-PQ intermediate path: `using ivf_pq::index_params nrows 10000, dim 128, n_lists 100, pq_dim 32`.

### Ownership

**hipVS / ROCm-DS on consumer gfx1100** — not Knowhere wiring.

Knowhere Catch2 recall 0.0 is consistent (broken/empty graph may not throw; search returns zeros).

**Phase B blocked** until library CAGRA build+search works on gfx1100.

### Next lab tries

`GRAPH_BUILD_ALGO=NN_DESCENT` on 2026-08-03 **did not apply** — RAFT still logged
`using ivf_pq::index_params` (Python IndexParams dropped/ignored the algo).

```bash
source ~/hipvs-bench-venv/bin/activate
export WORKDIR=~/rocmds_check_gfx1100
cd ~/ann-harness-amd && git pull --ff-only

# Must print: verify IndexParams.graph_build_algo=...  and NOT "using ivf_pq::"
MAX_TRAIN_ROWS=10000 MAX_QUERY_ROWS=200 ITOPK_SIZES=64 \
  GRAPH_BUILD_ALGO=NN_DESCENT \
  bash scripts/run_hipvs_cagra_bench.sh

# Quote globs (or pass many paths — newest wins)
python3 scripts/classify_cagra_triage.py \
  --hipvs-log "$WORKDIR/logs/cagra_hipvs_"*.log \
  --catch2-log "$WORKDIR/logs/cagra_catch2_20260803_224122.log"
```

If ctor cannot set NN_DESCENT → escalate to ROCm-DS (bindings + gfx1100 IVF_PQ graph).
If NN_DESCENT applies and still throws → same ownership, kernel/graph bug.
