#!/usr/bin/env bash
# Configure knowhere/build for Layer 2 (HIP / prebuilt hipVS).
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
KNOWHERE_DIR="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
BUILD_DIR="${KNOWHERE_DIR}/build"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

export INSTALL_PREFIX
export ROCM_PATH HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"
export PATH="${ROCM_PATH}/llvm/bin:${HOME}/.local/bin:${PATH}"

if ! command -v clang++ >/dev/null 2>&1; then
  echo "ERROR: clang++ not found (required for ROCm HIP host compile)." >&2
  echo "  Install rocm-dev and ensure ${ROCM_PATH}/llvm/bin is on PATH." >&2
  exit 1
fi

if ! grep -q 'Early WITH_HIP before project' "${KNOWHERE_DIR}/CMakeLists.txt" 2>/dev/null; then
  echo "ERROR: Layer 2 patches not applied under ${KNOWHERE_DIR}" >&2
  echo "  bash $(dirname "$0")/apply_knowhere_layer2_patches.sh ${KNOWHERE_DIR}" >&2
  exit 1
fi

if [ ! -f "${INSTALL_PREFIX}/lib/cmake/cuvs/cuvs-config.cmake" ]; then
  echo "ERROR: hipVS not found at ${INSTALL_PREFIX}/lib/cmake/cuvs/cuvs-config.cmake" >&2
  exit 1
fi

if [ ! -f "${BUILD_DIR}/Release/generators/conan_toolchain.cmake" ]; then
  echo "ERROR: run conan install first in ${BUILD_DIR} with -o with_cuvs=True" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Wipe stale cmake/make outputs but keep conan Release/generators.
# Deleting only CMakeFiles/ leaves Makefile rules pointing at missing
# VerifyGlobs.cmake (cmake_check_build_system fails on next build).
_conan_release="${BUILD_DIR}/Release"
if [ -d "${_conan_release}" ]; then
  _conan_tmp="$(mktemp -d)"
  cp -a "${_conan_release}" "${_conan_tmp}/Release"
fi
find "${BUILD_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
if [ -n "${_conan_tmp:-}" ]; then
  mkdir -p "${BUILD_DIR}/Release"
  cp -a "${_conan_tmp}/Release/." "${BUILD_DIR}/Release/"
  rm -rf "${_conan_tmp}"
fi
unset _conan_release _conan_tmp

cmake .. -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${BUILD_DIR}/Release/generators/conan_toolchain.cmake" \
  -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX};${ROCM_PATH}" \
  -DWITH_CUVS=ON \
  -DWITH_HIP=ON \
  "$@"

echo ""
echo "--- sanity ---"
grep -E 'WITH_CUVS:|WITH_HIP:|CMAKE_CXX_COMPILER:' CMakeCache.txt | head -5
grep -E 'pre-project CXX|WITH_HIP: raft|Configuring done' "${WORKDIR}/knowhere_cmake_hip.log" 2>/dev/null || true
