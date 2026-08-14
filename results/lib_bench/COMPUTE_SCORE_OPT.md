# IVF-PQ `ivfpq_compute_score` HIP opt (gfx1100)

## What changed

Patch: `patches/hipvs/0004-ivf-pq-compute-score-pipeline-pq8-gfx1100.patch`  
Apply: `scripts/apply_hipvs_ivf_pq_compute_score_opt.sh`

Inside `ivfpq_compute_score` (**HIP only**):

1. **Software-pipeline** interleaved PQ loads (`load nxt` while scoring `cur`)
2. **`pq_bits == 8` specialization** — treat each loaded byte as a code; skip the
   recursive bit-unpack `ivfpq_compute_chunk` (our default IVF_PQ recipe)

Non-HIP builds keep the stock loop.

## Lab commands

```bash
cd ~/ann-harness-amd && git pull
export WORKDIR=~/rocmds_check_gfx1100
bash scripts/apply_hipvs_ivf_pq_compute_score_opt.sh

cd $WORKDIR/hipVS
INSTALL_PREFIX=$WORKDIR/install ./build.sh libcuvs python \
  '--cmake-args="-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF"' \
  --gpu-arch=gfx1100
# reinstall python wheel into ~/hipvs-bench-venv

cd ~/ann-harness-amd
source ~/hipvs-bench-venv/bin/activate
INDEX_TYPE=IVF_PQ M=32 LUT_DTYPE=float32 P99_SAMPLE=0 \
  bash scripts/run_hipvs_ivf_bench.sh
```

Compare QPS to the previous `FILLED_pq_m32.md` baseline (~0.5× cuVS).  
Optional: search-only rocprof — fraction of time in `compute_similarity_kernel`.

## Status

| Date | Result |
|------|--------|
| (fill) | before / after hipVS QPS, nprobe grid, notes |
