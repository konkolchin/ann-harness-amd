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

### Catch2 (updated 2026-08-05)

After `0052` (`build_algo=NN_DESCENT` in `cagra_gen`):

| Assert | Section (stock line → +1) | Index | Result |
|--------|---------------------------|-------|--------|
| recall > 0.7 | **Search With Bitset** (:205→206) | `GPU_CUVS_CAGRA` | **0.0** |
| recall ≥ 0.85 | Search TopK (:242→243) | `GPU_CUVS_IVF_PQ` | near-miss (out of scope) |
| recall ≥ 0.8 | **Search Simple Bitset** (:328→329) | `GPU_CUVS_CAGRA` | **0.0** |
| recall ≥ 0.75 | main Search (:63) | `GPU_IVF_PQ` | near-miss |

Cosine / Hamming CAGRA green. Serialize unfiltered self-search OK (`id=i`, dist 0).  
ID dump (2026-08-05): bitset = wrong IDs; simple_bitset = all `-1`/`FLT_MAX`.

**Filter ownership (2026-08-05 hipVS Python, no Knowhere)**  
Log: `$WORKDIR/logs/lib_hipvs_cagra_filter_20260805_232922.{log,json}`

| Case | recall@1 | Notes |
|------|---------:|-------|
| unfiltered | **1.00** | self-ids correct |
| filter_40pct | **0.00** | wrong IDs, `neg1=0` (Catch2 bitset) |
| simple_bitset_64 | **0.00** | all `-1` (`neg1=64/64`, Catch2 simple bitset) |

**OWNER: hipVS CAGRA filtered search** on gfx1100.  
Escalation: [`docs/escalation_hipvs_cagra_filter_gfx1100.md`](../../docs/escalation_hipvs_cagra_filter_gfx1100.md) → https://github.com/ROCm-DS/hipVS/issues  

```bash
bash scripts/collect_hipvs_cagra_filter_escalation.sh   # env bundle to attach
# optional CUDA peer contrast:
bash scripts/run_cuvs_cagra_filter_repro.sh
```

Unfiltered path OK → Phase B smoke can proceed in parallel.

### IVF-PQ graph build (hipVS)

Still throws `graph_core` invalid/duplicated neighbors on gfx1100 — separate from nn_descent success.

## Next

```bash
# Full SIFT-1M lib (HDF5 GT valid) — both hosts
GRAPH_BUILD_ALGO=nn_descent GRAPH_DEGREE=64 INTERMEDIATE_GRAPH_DEGREE=128 \
  ITOPK_SIZES=64,128,256,512 bash scripts/run_*_cagra_bench.sh

# Then Catch2 + Phase B smoke once Knowhere CAGRA path uses a working build algo
```
