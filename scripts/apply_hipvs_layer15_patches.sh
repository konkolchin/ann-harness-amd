#!/usr/bin/env bash
# Apply Layer 1.5 hipVS patches (debug instrumentation + kIndexGroupSize warp-32 fix).
# Usage: bash scripts/apply_hipvs_layer15_patches.sh [path/to/hipVS]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPVS_DIR="${1:-${HOME}/rocmds_check_gfx1100/hipVS}"
PATCH_DIR="${REPO_ROOT}/patches/hipvs"

apply_patch() {
  local patch="$1"
  local name
  name="$(basename "$patch")"
  if git apply --reverse --check "$patch" 2>/dev/null; then
    echo "Already applied: ${name}"
    return 0
  fi
  git apply "$patch"
  echo "Applied: ${name}"
}

if [ ! -d "${HIPVS_DIR}/.git" ]; then
  echo "hipVS git checkout not found: ${HIPVS_DIR}" >&2
  exit 1
fi

cd "${HIPVS_DIR}"
apply_patch "${PATCH_DIR}/0001-ivf-packer-debug-instrumentation.patch"
apply_patch "${PATCH_DIR}/0002-ivf-kIndexGroupSize-warp32-gfx1100.patch"

echo ""
echo "Layer 1.5 patches applied under ${HIPVS_DIR}"
echo "Next: clean-rebuild (see docs/porting_milvus_gpu_to_amd.tex Step 4), then:"
echo "  bash scripts/check_ivf_packer_mismatch.sh --run"
