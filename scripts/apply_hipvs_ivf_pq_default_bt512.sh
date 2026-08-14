#!/usr/bin/env bash
# Hardcode AMD IVF-PQ compute_similarity blockDim prefer=512 (DXC gfx1100).
#
# Measured +28–36% library QPS vs stock auto-shrink (LAUNCH_KNOBS.md).
#
# Usage:
#   bash scripts/apply_hipvs_ivf_pq_default_bt512.sh [path/to/hipVS]
#
# Handles:
#   - clean hipVS → applies patch 0006
#   - tree with launch knobs (0005) → defaults force_block_threads to 512
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPVS_ROOT="${1:-${WORKDIR:-$HOME/rocmds_check_gfx1100}/hipVS}"
F="$HIPVS_ROOT/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh"
PATCH="${REPO_ROOT}/patches/hipvs/0006-ivf-pq-default-blockdim-512-amd.patch"

if [[ ! -f "$F" ]]; then
  echo "ERROR: missing $F" >&2
  exit 1
fi

if grep -q 'kAmdIvfPqPreferThreads' "$F"; then
  echo "OK: AMD prefer-512 already present in $F"
  exit 0
fi

# If patch 0005 (env knobs) is present, default force_block_threads to 512.
if grep -q 'uint32_t force_block_threads = 0;' "$F"; then
  if grep -q 'ann-harness gfx1100: launch A/B knobs' "$F"; then
    sed -i 's/uint32_t force_block_threads = 0;/uint32_t force_block_threads = 512;  \/\/ DXC gfx1100 default (LAUNCH_KNOBS.md)/' "$F"
    echo "OK: set force_block_threads default=512 (on top of launch knobs) in $F"
    exit 0
  fi
fi

if grep -q 'uint32_t force_block_threads = 512' "$F"; then
  echo "OK: force_block_threads already defaults to 512 in $F"
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
echo "Next: rebuild hipVS libcuvs (+ python if used), reinstall into Milvus/Knowhere prefix,"
echo "then restart dashboard Milvus so it loads the new libcuvs.so."
