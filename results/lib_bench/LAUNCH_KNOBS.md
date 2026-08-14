# IVF-PQ launch knobs (spots 6–7) — gfx1100 A/B

## What changed

Patch: `patches/hipvs/0005-ivf-pq-force-launch-knobs-gfx1100.patch`  
Apply: `scripts/apply_hipvs_ivf_pq_force_launch_knobs.sh`  
Sweep: `scripts/run_hipvs_ivf_pq_launch_knob_sweep.sh`  
Summarize: `python3 scripts/summarize_lib_bench_json.py …`

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
# rebuild libcuvs + python (Spot-1 recipe), then:
bash scripts/run_hipvs_ivf_pq_launch_knob_sweep.sh
python3 scripts/summarize_lib_bench_json.py \
  $WORKDIR/logs/launch_knobs/lib_hipvs_ivf_pq_m32_*.json
```

## Measured result (2026-08-14)

SIFT-1M, `nlist=1024`, `m=32`, `nbits=8`, `lut_dtype=float32`, `k=10`, RX 7900 XTX.  
Logs: `$WORKDIR/logs/launch_knobs/lib_hipvs_ivf_pq_m32_*_20260814_2024*.json`

| Case | nprobe=1 | nprobe=8 | nprobe=16 | nprobe=32 | vs baseline @32 | R@10 @32 |
|------|---------:|---------:|----------:|----------:|----------------:|---------:|
| baseline (auto) | 1,582,112 | 713,712 | 374,216 | 205,653 | — | 0.726 |
| fast | 1,550,780 | 714,306 | 375,133 | 205,161 | ~0% | 0.726 |
| no_basediff | 1,516,426 | 672,389 | 351,914 | 192,970 | −6% | 0.726 |
| no_smem_lut | 1,298,094 | 454,644 | 247,286 | 137,920 | −33% | 0.724 |
| **fast + bt256** | 1,521,352 | 747,162 | 412,594 | 224,774 | **+9%** | 0.726 |
| **fast + bt512** | 1,741,224 | 914,655 | 510,104 | **274,531** | **+34%** | 0.727 |
| **no_smem + bt256** | 1,753,696 | 910,772 | 519,546 | **273,103** | **+33%** | 0.724 |

Recall@10 flat across the grid (~0.34 → ~0.73).

### Interpretation

1. **`FORCE_VARIANT=fast` ≈ baseline** → stock picker already chooses `(Precomp,SMemLUT)`.
2. **`no_basediff` / bare `no_smem_lut`** hurt (especially global LUT with auto threads).
3. **Real lever = `blockDim`:** `HIPVS_IVF_PQ_BLOCK_THREADS=512` with `fast` gives **~+28–36% QPS** at `nprobe≥8`, same recall. `no_smem_lut` + `bt256` matches that — stock thread shrink for that variant is wrong on RDNA3.
4. Still **not** closing the full ~2× cuVS gap alone, but first concrete PQ search win on gfx1100.

**Verdict:** spots 6–7 **partial win** via larger block size. Next: bake a gfx1100-friendly default (prefer `n_threads≥512` when occupancy allows, or ship env default in bench scripts), optionally sweep `bt1024` + carveout; LUT-build (spot 5) remains secondary.

## Re-check best case

```bash
HIPVS_IVF_PQ_FORCE_VARIANT=fast HIPVS_IVF_PQ_BLOCK_THREADS=512 \
  INDEX_TYPE=IVF_PQ M=32 LUT_DTYPE=float32 P99_SAMPLE=0 \
  bash scripts/run_hipvs_ivf_bench.sh
```
