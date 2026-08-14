#!/usr/bin/env bash
# REVERT Spot-1: remove ivfpq_compute_score pipeline / pq8 HIP opt (no QPS win).
#
# Usage:
#   bash scripts/revert_hipvs_ivf_pq_compute_score_opt.sh [path/to/hipVS]
#
# Then rebuild libcuvs (+ python) and restart Milvus if the product path links
# this install.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPVS_ROOT="${1:-${WORKDIR:-$HOME/rocmds_check_gfx1100}/hipVS}"
F="$HIPVS_ROOT/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh"
PATCH="${REPO_ROOT}/patches/hipvs/0004-ivf-pq-compute-score-pipeline-pq8-gfx1100.patch"

if [[ ! -f "$F" ]]; then
  echo "ERROR: missing $F" >&2
  exit 1
fi

if ! grep -q 'gfx1100 tune (ann-harness): software-pipeline' "$F"; then
  echo "OK: Spot-1 compute_score opt not present in $F (nothing to revert)"
  exit 0
fi

if [[ ! -f "$PATCH" ]]; then
  echo "ERROR: missing $PATCH (needed for reverse apply)" >&2
  exit 1
fi

if [[ -d "$HIPVS_ROOT/.git" ]]; then
  if (cd "$HIPVS_ROOT" && git apply -R --check "$PATCH" 2>/dev/null); then
    (cd "$HIPVS_ROOT" && git apply -R "$PATCH")
    echo "Reverted via git apply -R: $(basename "$PATCH")"
  else
    echo "NOTE: git apply -R --check failed; trying patch -R -p1" >&2
    (cd "$HIPVS_ROOT" && patch -R -p1 < "$PATCH")
    echo "Reverted via patch -R -p1: $(basename "$PATCH")"
  fi
else
  (cd "$HIPVS_ROOT" && patch -R -p1 < "$PATCH")
  echo "Reverted via patch -R -p1: $(basename "$PATCH")"
fi

if grep -q 'gfx1100 tune (ann-harness): software-pipeline' "$F"; then
  echo "ERROR: marker still present after revert — fix $F manually" >&2
  exit 1
fi

echo ""
echo "Spot-1 removed from sources. Rebuild hipVS before measuring:"
echo "  cd \"$HIPVS_ROOT\""
echo "  INSTALL_PREFIX=\$WORKDIR/install ./build.sh libcuvs python \\"
echo "    '--cmake-args=\"-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF\"' \\"
echo "    --gpu-arch=gfx1100"
echo "  # then reinstall python/cuvs if needed; restart HIP Milvus for product path"
