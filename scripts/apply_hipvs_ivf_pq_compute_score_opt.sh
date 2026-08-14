#!/usr/bin/env bash
# Optimize hot ivfpq_compute_score on HIP (gfx1100 / RDNA3).
#
# Changes (HIP-only, inside compute_similarity_kernel path):
#   1) Software-pipeline interleaved PQ vector loads (hide global latency)
#   2) Specialize pq_bits==8: byte LUT gathers, skip recursive bit-unpack
#
# Usage:
#   bash scripts/apply_hipvs_ivf_pq_compute_score_opt.sh [path/to/hipVS]
#
# Then rebuild hipVS libcuvs (+ python) and re-bench:
#   INDEX_TYPE=IVF_PQ M=32 P99_SAMPLE=0 bash scripts/run_hipvs_ivf_bench.sh
# Optional: search-only rocprof — expect lower % in compute_similarity_kernel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPVS_ROOT="${1:-${WORKDIR:-$HOME/rocmds_check_gfx1100}/hipVS}"
F="$HIPVS_ROOT/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh"
PATCH="${REPO_ROOT}/patches/hipvs/0004-ivf-pq-compute-score-pipeline-pq8-gfx1100.patch"

if [[ ! -f "$F" ]]; then
  echo "ERROR: missing $F" >&2
  exit 1
fi

if grep -q 'gfx1100 tune (ann-harness): software-pipeline' "$F"; then
  echo "OK: compute_score HIP opt already present in $F"
  exit 0
fi

if [[ ! -f "$PATCH" ]]; then
  echo "ERROR: missing $PATCH" >&2
  exit 1
fi

if [[ -d "$HIPVS_ROOT/.git" ]]; then
  if (cd "$HIPVS_ROOT" && git apply --check "$PATCH" 2>/dev/null); then
    (cd "$HIPVS_ROOT" && git apply "$PATCH")
    echo "Applied via git: $(basename "$PATCH")"
  else
    echo "NOTE: git apply --check failed; trying patch -p1" >&2
    (cd "$HIPVS_ROOT" && patch -p1 < "$PATCH")
    echo "Applied via patch -p1: $(basename "$PATCH")"
  fi
else
  (cd "$HIPVS_ROOT" && patch -p1 < "$PATCH")
  echo "Applied via patch -p1: $(basename "$PATCH")"
fi

echo ""
echo "Next (AMD lab):"
echo "  cd \"$HIPVS_ROOT\""
echo "  INSTALL_PREFIX=\$WORKDIR/install ./build.sh libcuvs python \\"
echo "    '--cmake-args=\"-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF\"' \\"
echo "    --gpu-arch=gfx1100"
echo "  # reinstall into ~/hipvs-bench-venv, then:"
echo "  INDEX_TYPE=IVF_PQ M=32 LUT_DTYPE=float32 P99_SAMPLE=0 \\"
echo "    bash scripts/run_hipvs_ivf_bench.sh"
