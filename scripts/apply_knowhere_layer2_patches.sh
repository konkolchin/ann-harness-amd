#!/usr/bin/env bash
# Apply Layer 2 Knowhere patches (HIP CMake + GPU device shim).
# Usage: bash scripts/apply_knowhere_layer2_patches.sh [path/to/knowhere]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KNOWHERE_DIR="${1:-${HOME}/rocmds_check_gfx1100/knowhere}"
PATCH_DIR="${REPO_ROOT}/patches/knowhere"

apply_patch() {
  local patch="$1"
  local name
  name="$(basename "$patch")"

  if git apply --check "$patch" 2>/dev/null; then
    git apply "$patch"
    echo "Applied: ${name}"
    return 0
  fi

  if git apply --reverse --check "$patch" 2>/dev/null; then
    echo "Already applied: ${name}"
    return 0
  fi

  echo "ERROR: cannot apply ${name} (tree is not clean 2.5 + prior patches)." >&2
  echo "  cd ${KNOWHERE_DIR} && git checkout -- . && bash $0 ${KNOWHERE_DIR}" >&2
  exit 1
}

if [ ! -d "${KNOWHERE_DIR}/.git" ]; then
  echo "knowhere git checkout not found: ${KNOWHERE_DIR}" >&2
  exit 1
fi

cd "${KNOWHERE_DIR}"
apply_patch "${PATCH_DIR}/0001-cmake-with-hip-prebuilt-hipvs.patch"
apply_patch "${PATCH_DIR}/0002-gpu-device-count-hip-shim.patch"
apply_patch "${PATCH_DIR}/0003-libhipcuvs-find-rocm-hip.patch"
apply_patch "${PATCH_DIR}/0004-strip-hip-offload-host-cxx.patch"
apply_patch "${PATCH_DIR}/0005-host-cxx-no-offload-arch.patch"
apply_patch "${PATCH_DIR}/0006-gxx-wrapper-strip-offload-arch.patch"
apply_patch "${PATCH_DIR}/0007-preproject-cxx-wrapper-and-launch-filter.patch"
apply_patch "${PATCH_DIR}/0008-add-libhipcuvs-preproject.cmake.patch"
apply_patch "${PATCH_DIR}/0009-early-with-hip-auto-detect.patch"

echo ""
echo "Layer 2 patches applied under ${KNOWHERE_DIR}"
echo "Next:"
echo "  export INSTALL_PREFIX=~/rocmds_check_gfx1100/install"
echo "  cmake .. -DWITH_HIP=ON  (optional if INSTALL_PREFIX has hipVS; auto-detected by 0009)"
echo "  See porting runbook Layer 2."
