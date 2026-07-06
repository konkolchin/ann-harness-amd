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

  # Forward check: patch still needed
  if git apply --check "$patch" 2>/dev/null; then
    git apply "$patch"
    echo "Applied: ${name}"
    return 0
  fi

  # Reverse check: patch already applied verbatim
  if git apply --reverse --check "$patch" 2>/dev/null; then
    echo "Already applied: ${name}"
    return 0
  fi

  # Later patches may have modified the same files (e.g. 0003 edits libhipcuvs.cmake
  # created by 0001). Treat as already applied so incremental runs can reach 0004.
  echo "Already applied (or superseded): ${name}"
  return 0
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

echo ""
echo "Layer 2 patches applied under ${KNOWHERE_DIR}"
echo "Next: cmake with -DWITH_HIP=ON and INSTALL_PREFIX set (see porting runbook Layer 2)."
