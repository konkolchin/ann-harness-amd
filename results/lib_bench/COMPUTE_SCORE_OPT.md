# IVF-PQ `ivfpq_compute_score` HIP opt (gfx1100)

## What changed

Patch: `patches/hipvs/0004-ivf-pq-compute-score-pipeline-pq8-gfx1100.patch`  
Apply: `scripts/apply_hipvs_ivf_pq_compute_score_opt.sh`

Inside `ivfpq_compute_score` (**HIP only**):

1. **Software-pipeline** interleaved PQ loads (`load nxt` while scoring `cur`)
2. **`pq_bits == 8` specialization** — treat each loaded byte as a code; skip the
   recursive bit-unpack `ivfpq_compute_chunk` (our default IVF_PQ recipe)

Non-HIP builds keep the stock loop.

## Lab rebuild notes (gotchas)

`hipVS/build.sh` + Python/`skbuild` on this host:

- Wipe `cpp/build` if CMake still has Instinct `GPU_TARGETS`
- Put `'--cmake-args="..."'` **before** `--gpu-arch=gfx1100` (greedy arch parser)
- Set `ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0` (avoid compiling for iGPU `gfx1036`)
- For pip wheels, use **semicolon-separated** `SKBUILD_CMAKE_ARGS` including
  `-DUSE_WARPSIZE_32=ON` and `-DCMAKE_HIP_ARCHITECTURES=gfx1100`
- Install `./python/cuvs` with `--no-deps` so pip does not pull PyPI `amd-libhipvs==0.0.2`

## Measured result (2026-08-14)

Same recipe as `FILLED_pq_m32.md`: SIFT-1M, `nlist=1024`, `m=32`, `nbits=8`,
`lut_dtype=float32`, `k=10`, RX 7900 XTX / gfx1100.

JSON: `~/rocmds_check_gfx1100/logs/lib_hipvs_ivf_pq_m32_20260814_200854.json`

| nprobe | Baseline hipVS QPS (2026-07-23) | After opt QPS | Δ |
|-------:|--------------------------------:|--------------:|---|
| 1 | 1,593,707 | 1,596,881 | ~0% |
| 4 | 1,268,942 | 1,072,488 | down |
| 8 | 710,989 | 720,729 | ~+1% |
| 16 | 393,742 | 375,005 | slight down |
| 32 | 217,275 | 205,738 | slight down |

Recall@10 unchanged (~0.35 → ~0.73 across the grid).

**Verdict:** no meaningful QPS win. Spot‑1 (`ivfpq_compute_score` pipeline / pq8
byte path) is **not** the lever for the ~2× gap vs cuVS. Prefer next deck items:
launch/carveout heuristics (spots 6–7) or LUT-build specialization (spot 5).

## Re-bench commands

```bash
cd ~/ann-harness-amd && git pull
export WORKDIR=~/rocmds_check_gfx1100
export ROCM_PATH=/opt/rocm-7.0.2
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
source ~/hipvs-bench-venv/bin/activate

INDEX_TYPE=IVF_PQ M=32 LUT_DTYPE=float32 P99_SAMPLE=0 \
  bash scripts/run_hipvs_ivf_bench.sh
```
