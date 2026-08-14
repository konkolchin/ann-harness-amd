# IVF-PQ `ivfpq_compute_score` HIP opt (gfx1100) — **REVERTED**

## Status

**Reverted 2026-08-14.** Measured: no meaningful QPS win (flat / slight down).
Do **not** re-apply. Historical patch kept only as a record:

- Patch (archive): `patches/hipvs/0004-ivf-pq-compute-score-pipeline-pq8-gfx1100.patch`
- Apply script: **refuses** (`scripts/apply_hipvs_ivf_pq_compute_score_opt.sh`)
- Revert lab tree: `scripts/revert_hipvs_ivf_pq_compute_score_opt.sh`

What it tried (HIP-only inside `ivfpq_compute_score`):

1. Software-pipeline interleaved PQ loads
2. `pq_bits == 8` byte LUT path (skip bit-unpack)

## Measured result (before revert, 2026-08-14)

Same recipe as `FILLED_pq_m32.md`: SIFT-1M, `m=32`, `nbits=8`, RX 7900 XTX.

| nprobe | Baseline hipVS QPS (2026-07-23) | After opt QPS | Δ |
|-------:|--------------------------------:|--------------:|---|
| 1 | 1,593,707 | 1,596,881 | ~0% |
| 4 | 1,268,942 | 1,072,488 | down |
| 8 | 710,989 | 720,729 | ~+1% |
| 16 | 393,742 | 375,005 | slight down |
| 32 | 217,275 | 205,738 | slight down |

Recall@10 unchanged.

**Verdict:** miss. Real lever was launch `blockDim=512` (`LAUNCH_KNOBS.md` / patch `0006`).

## Lab: strip Spot-1 if still in your hipVS tree

```bash
cd ~/ann-harness-amd && git pull
export WORKDIR=~/rocmds_check_gfx1100
bash scripts/revert_hipvs_ivf_pq_compute_score_opt.sh "$WORKDIR/hipVS"
# rebuild libcuvs (+ python); restart Milvus if product path uses this install
```
