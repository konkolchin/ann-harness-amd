# Library hipVS vs cuVS — IVF_FLAT heavy nprobe (2026-07-27)

GPU-heavy library microbench (no Milvus): SIFT-1M, `nlist=1024`, `k=10`,
`nprobe ∈ {32,64,128,256}`. Shows compute-bound search (QPS ~halves as nprobe doubles).

| nprobe | hipVS QPS (7900) | cuVS QPS (4080) | R@10 | speed-up (hip/cu) |
|--------|------------------|-----------------|------|-------------------|
| 32 | 64,142 | 49,039 | 0.978 | **1.31×** |
| 64 | 33,381 | 24,759 | 0.996 | **1.35×** |
| 128 | 17,200 | 12,498 | 0.999 | **1.38×** |
| 256 | 8,784 | 6,332 | 0.999 | **1.39×** |

Sources:
- `lib_hipvs_ivf_flat_heavy_20260727_195243.json`
- `lib_cuvs_ivf_flat_heavy_20260727_200221.json`

Contrast: library IVF_PQ m=32 was ~0.5× (see `FILLED_pq_m32.md`). Under this FLAT/high-nprobe load, AMD is ~1.3–1.4× NVIDIA on peer GPUs.
