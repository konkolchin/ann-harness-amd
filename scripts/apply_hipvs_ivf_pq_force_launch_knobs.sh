#!/usr/bin/env bash
# Add env knobs to force hipVS IVF-PQ compute_similarity_select launch variants
# (spots 6–7 in the opt deck). Host-side only; score math unchanged.
#
# Env (after rebuild + reinstall python wheel):
#   HIPVS_IVF_PQ_FORCE_VARIANT=fast|no_basediff|no_smem_lut
#   HIPVS_IVF_PQ_FORCE_SMEM_LUT=0|1   # with FORCE_PRECOMP
#   HIPVS_IVF_PQ_FORCE_PRECOMP=0|1
#   HIPVS_IVF_PQ_BLOCK_THREADS=128|256|512|1024
#   HIPVS_IVF_PQ_FORCE_NO_LOCAL_TOPK=1
#   HIPVS_IVF_PQ_PREFERRED_CARVEOUT=0.0..1.0
#
# Usage:
#   bash scripts/apply_hipvs_ivf_pq_force_launch_knobs.sh [path/to/hipVS]
#
# Safe on top of patch 0004 (compute_score opt); both touch the same file but
# different regions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPVS_ROOT="${1:-${WORKDIR:-$HOME/rocmds_check_gfx1100}/hipVS}"
F="$HIPVS_ROOT/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh"
PATCH="${REPO_ROOT}/patches/hipvs/0005-ivf-pq-force-launch-knobs-gfx1100.patch"

if [[ ! -f "$F" ]]; then
  echo "ERROR: missing $F" >&2
  exit 1
fi

if grep -q 'ann-harness gfx1100: launch A/B knobs' "$F"; then
  echo "OK: launch knobs already present in $F"
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
echo "Next (AMD lab): rebuild hipVS (wipe cpp/build if stale), reinstall ./python/cuvs --no-deps,"
echo "then A/B with scripts/run_hipvs_ivf_pq_launch_knob_sweep.sh"
echo "  or single:"
echo "  HIPVS_IVF_PQ_FORCE_VARIANT=fast HIPVS_IVF_PQ_BLOCK_THREADS=256 \\"
echo "    INDEX_TYPE=IVF_PQ M=32 P99_SAMPLE=0 bash scripts/run_hipvs_ivf_bench.sh"
