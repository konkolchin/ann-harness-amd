# IVF-PQ launch knobs (spots 6–7) — gfx1100 A/B

## What changed

Patch: `patches/hipvs/0005-ivf-pq-force-launch-knobs-gfx1100.patch`  
Apply: `scripts/apply_hipvs_ivf_pq_force_launch_knobs.sh`  
Sweep: `scripts/run_hipvs_ivf_pq_launch_knob_sweep.sh`

Host-side short-circuit of `compute_similarity_select` via env (no score math change).
On first search with knobs set, stderr prints:

```text
[ann-harness] ivf_pq compute_similarity_select: variant=... block_threads=... ...
```

| Env | Values | Effect |
|-----|--------|--------|
| `HIPVS_IVF_PQ_FORCE_VARIANT` | `fast` / `no_basediff` / `no_smem_lut` | Force `(Precomp,SMemLUT)` template |
| `HIPVS_IVF_PQ_FORCE_SMEM_LUT` + `FORCE_PRECOMP` | `0`/`1` | Same mapping without VARIANT name |
| `HIPVS_IVF_PQ_BLOCK_THREADS` | power-of-two ≥32 | Fix `blockDim.x` (skip auto shrink) |
| `HIPVS_IVF_PQ_PREFERRED_CARVEOUT` | `0.0`–`1.0` | Override preferred shmem carveout |

**Do not use** `HIPVS_IVF_PQ_FORCE_NO_LOCAL_TOPK=1` alone: it only flips
`manage_local_topk` inside `compute_similarity_select`, while the IVF-PQ search
caller still allocates/reads fused local top-k outputs → **hipErrorIllegalMemoryAccess (700)**
on gfx1100 (seen 2026-08-14). Proper fused-topk off needs a matching search-path change.

Compatible with patch **0004** (compute_score pipeline); apply either order.

## Lab steps

```bash
cd ~/ann-harness-amd && git pull
export WORKDIR=~/rocmds_check_gfx1100
export ROCM_PATH=/opt/rocm-7.0.2
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0

bash scripts/apply_hipvs_ivf_pq_force_launch_knobs.sh "$WORKDIR/hipVS"

cd "$WORKDIR/hipVS"
rm -rf cpp/build   # if CMake still has wrong GPU_TARGETS
INSTALL_PREFIX=$WORKDIR/install ./build.sh libcuvs python \
  '--cmake-args="-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF"' \
  --gpu-arch=gfx1100

source ~/hipvs-bench-venv/bin/activate
# reinstall python/cuvs with --no-deps + SKBUILD_CMAKE_ARGS as in COMPUTE_SCORE_OPT.md

cd ~/ann-harness-amd
# quick single:
HIPVS_IVF_PQ_FORCE_VARIANT=fast HIPVS_IVF_PQ_BLOCK_THREADS=256 \
  INDEX_TYPE=IVF_PQ M=32 LUT_DTYPE=float32 P99_SAMPLE=0 \
  bash scripts/run_hipvs_ivf_bench.sh

# or full grid:
bash scripts/run_hipvs_ivf_pq_launch_knob_sweep.sh
```

## Measured result

_(fill after lab)_

| Case | nprobe=1 QPS | nprobe=8 | nprobe=32 | Recall@10 |
|------|-------------:|---------:|----------:|----------:|
| baseline (auto) | | | | |
| fast | | | | |
| no_basediff | | | | |
| no_smem_lut | | | | |
| fast + bt256 | | | | |
| fast + bt512 | | | | |
| no_smem + bt256 | | | | |
| no_local_topk | **CRASH 700** — unsafe knob; skip | | | |

After a sweep on the lab:

```bash
python3 scripts/compare_cuvs_lib_json.py \
  $WORKDIR/logs/launch_knobs/lib_hipvs_ivf_pq_m32_*.json
```

**Verdict:** _(TBD — look for ≥5–10% QPS move at nprobe≥8 with unchanged recall)_
