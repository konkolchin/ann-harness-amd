# CAGRA lib / Catch2 status (gfx1100)

## Phase A — library (updated 2026-08-04)

### Fair smoke (10k train / 200 query, exact GT, `nn_descent`, deg 64/128)

| itopk | cuVS 4080 QPS | hipVS gfx1100 QPS | hip/cu | recall@10 |
|------:|-------------:|------------------:|-------:|----------:|
| 64 | 418365 | 142567 | **0.34×** | **1.00 / 1.00** |
| 128 | 237180 | 78075 | **0.33×** | **1.00 / 1.00** |
| 256 | 114756 | 46882 | **0.41×** | **1.00 / 1.00** |
| 512 | 56116 | 1880 | **0.03×** | **1.00 / 1.00** |

JSON:
- CUDA: `~/milvus_cuda_4080/logs/lib_cuvs_cagra_20260804_220649.json`
- AMD: `$WORKDIR/logs/lib_hipvs_cagra_20260804_220816.json`

**Verdict:** consumer gfx1100 **can** run CAGRA (`nn_descent`) with correct recall.  
IVF-PQ graph build still throws on hipVS; use `build_algo=nn_descent`.  
Note AMD QPS cliff at itopk=512 (investigate later; recall still 1.0).

### Earlier false alarm

`recall@10=0.0695` flat on **both** CUDA and AMD was invalid GT (HDF5 neighbors vs full 1M while indexing 10k). Fixed in `bench_cuvs_cagra.py` (`gt_source=exact_subset`).

### Catch2 (still open)

Full `Test All GPU Index` (2026-08-03): CAGRA recall **0.0** at TopK / Serialize — likely default **ivf_pq** graph build inside Knowhere. Next: confirm Knowhere CAGRA build algo; prefer nn_descent or fix ivf_pq path.

### IVF-PQ graph build (hipVS)

Still throws `graph_core` invalid/duplicated neighbors on gfx1100 — separate from nn_descent success.

## Next

```bash
# Full SIFT-1M lib (HDF5 GT valid) — both hosts
GRAPH_BUILD_ALGO=nn_descent GRAPH_DEGREE=64 INTERMEDIATE_GRAPH_DEGREE=128 \
  ITOPK_SIZES=64,128,256,512 bash scripts/run_*_cagra_bench.sh

# Then Catch2 + Phase B smoke once Knowhere CAGRA path uses a working build algo
```
