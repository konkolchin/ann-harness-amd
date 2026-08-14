#!/usr/bin/env bash
# DEPRECATED — Spot-1 gave no QPS win (2026-08-14). Do not apply.
# Use: bash scripts/revert_hipvs_ivf_pq_compute_score_opt.sh
set -euo pipefail
echo "REFUSED: Spot-1 ivfpq_compute_score pipeline/pq8 opt is reverted (no QPS win)." >&2
echo "  See results/lib_bench/COMPUTE_SCORE_OPT.md" >&2
echo "  To remove it from a lab tree:" >&2
echo "    bash scripts/revert_hipvs_ivf_pq_compute_score_opt.sh [path/to/hipVS]" >&2
exit 1
