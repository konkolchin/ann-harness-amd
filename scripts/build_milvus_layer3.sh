#!/usr/bin/env bash
# Build Milvus v2.5.4 Standalone with HIP Knowhere (Layer 3).
#
# Usage (on amd-rx7900xtx, inside tmux):
#   bash scripts/build_milvus_layer3.sh
#   SKIP_CLONE=1 bash scripts/build_milvus_layer3.sh   # reuse existing tree
#
# Prefer local HIP Knowhere (Layer 2 tree) via MILVUS_KNOWHERE_SOURCE_DIR.
# Falls back to FetchContent from DXC knowhere 2.5.
#
# Logs: $WORKDIR/milvus_layer3_build.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
MILVUS_DIR="${MILVUS_DIR:-${WORKDIR}/milvus}"
KNOWHERE_DIR="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
MILVUS_TAG="${MILVUS_TAG:-v2.5.4}"
LOG="${WORKDIR}/milvus_layer3_build.log"
SHIM_DIR="${SHIM_DIR:-${WORKDIR}/libshims}"

# hipVS/xxHash live under INSTALL_PREFIX. Milvus core_build.sh overwrites
# INSTALL_PREFIX for its own output tree — keep the ROCmDS prefix separately.
export MILVUS_HIP_INSTALL_PREFIX="${MILVUS_HIP_INSTALL_PREFIX:-${INSTALL_PREFIX}}"
export ROCMDS_INSTALL_PREFIX="${ROCMDS_INSTALL_PREFIX:-${INSTALL_PREFIX}}"
export INSTALL_PREFIX ROCM_PATH HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"
export PATH="${ROCM_PATH}/llvm/bin:${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${MILVUS_HIP_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
# core_build.sh defaults CUDA_COMPILER to /usr/local/cuda/bin/nvcc; AMD HIP builds
# must not require nvcc (src CMakeLists is patched to skip CUDA language).
export CUDA_COMPILER="${CUDA_COMPILER:-$(command -v hipcc 2>/dev/null || echo /bin/true)}"

if [ ! -f "${INSTALL_PREFIX}/lib/cmake/cuvs/cuvs-config.cmake" ]; then
  echo "ERROR: hipVS not found at ${INSTALL_PREFIX}/lib/cmake/cuvs/" >&2
  echo "  Complete Layer 1 / 1.5 first." >&2
  exit 1
fi

# Knowhere libfaiss.cmake needs CONFIG-mode xxHash. Use a *standalone* install under
# MILVUS_HIP_INSTALL_PREFIX — do NOT use Knowhere Conan CMakeDeps generators
# (xxHashConfig.cmake there calls check_build_type_defined and breaks Milvus configure).
if [ -f "${MILVUS_HIP_INSTALL_PREFIX}/lib/cmake/xxHash/xxHashConfig.cmake" ] || \
   [ -f "${MILVUS_HIP_INSTALL_PREFIX}/lib64/cmake/xxHash/xxHashConfig.cmake" ]; then
  echo "==> xxHash already in MILVUS_HIP_INSTALL_PREFIX"
else
  echo "==> installing xxHash into ${MILVUS_HIP_INSTALL_PREFIX} (standalone, not Conan)"
  _xx_src="${WORKDIR}/src/xxHash"
  mkdir -p "${WORKDIR}/src"
  if [ ! -d "${_xx_src}/.git" ]; then
    git clone --depth 1 --branch v0.8.2 https://github.com/Cyan4973/xxHash.git "${_xx_src}"
  fi
  cmake -S "${_xx_src}/cmake_unofficial" -B "${_xx_src}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${MILVUS_HIP_INSTALL_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON
  cmake --build "${_xx_src}/build" -j"$(nproc)"
  cmake --install "${_xx_src}/build"
  if [ ! -f "${MILVUS_HIP_INSTALL_PREFIX}/lib/cmake/xxHash/xxHashConfig.cmake" ] && \
     [ ! -f "${MILVUS_HIP_INSTALL_PREFIX}/lib64/cmake/xxHash/xxHashConfig.cmake" ]; then
    echo "ERROR: xxHashConfig.cmake missing after install under ${MILVUS_HIP_INSTALL_PREFIX}" >&2
    exit 1
  fi
fi
# Never prepend Knowhere Conan generators — they poison find_package(xxHash).
export CMAKE_PREFIX_PATH="${MILVUS_HIP_INSTALL_PREFIX};${ROCM_PATH};${CMAKE_PREFIX_PATH:-}"
# Explicitly clear so milvus Knowhere CMakeLists does not pick Conan xxHash.
export KNOWHERE_CONAN_GENERATORS=""

if [ "${SKIP_CLONE:-0}" != "1" ]; then
  if [ ! -d "${MILVUS_DIR}/.git" ]; then
    echo "==> clone milvus ${MILVUS_TAG}"
    git clone -b "${MILVUS_TAG}" --depth 1 https://github.com/milvus-io/milvus.git "${MILVUS_DIR}"
  else
    echo "==> milvus already present: ${MILVUS_DIR}"
  fi
fi

echo "==> apply Layer 3 patches"
bash "${REPO_ROOT}/scripts/apply_milvus_layer3_patches.sh" "${MILVUS_DIR}"

# Prefer the already-patched Layer-2 knowhere tree on the lab host.
CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS:-}"
if [ -f "${KNOWHERE_DIR}/CMakeLists.txt" ] && grep -q 'Early WITH_HIP before project' "${KNOWHERE_DIR}/CMakeLists.txt" 2>/dev/null; then
  echo "==> using local HIP Knowhere: ${KNOWHERE_DIR}"
  CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DMILVUS_KNOWHERE_SOURCE_DIR=${KNOWHERE_DIR}"
else
  echo "==> FetchContent will pull DXC Knowhere 2.5 (set KNOWHERE_DIR to override)"
fi
# Force standalone xxHash_DIR (never Knowhere Conan generators).
_xx_cfg=""
for _c in \
  "${MILVUS_HIP_INSTALL_PREFIX}/lib/cmake/xxHash/xxHashConfig.cmake" \
  "${MILVUS_HIP_INSTALL_PREFIX}/lib64/cmake/xxHash/xxHashConfig.cmake"
do
  if [ -f "${_c}" ]; then _xx_cfg="${_c}"; break; fi
done
if [ -n "${_xx_cfg}" ]; then
  CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DxxHash_DIR=$(dirname "${_xx_cfg}")"
  echo "==> CMAKE xxHash_DIR=$(dirname "${_xx_cfg}") (standalone)"
else
  echo "ERROR: standalone xxHashConfig.cmake not found under ${MILVUS_HIP_INSTALL_PREFIX}" >&2
  exit 1
fi
export CMAKE_EXTRA_ARGS

echo "==> install Milvus build deps (idempotent)"
if [ -x "${MILVUS_DIR}/scripts/install_deps.sh" ]; then
  bash "${MILVUS_DIR}/scripts/install_deps.sh" || true
fi

# Stale/incomplete cmake_build (failed configure) has no 'install' target; core_build.sh
# still runs `make install` and reports a misleading "No rule to make target".
_cmake_build="${MILVUS_DIR}/cmake_build"
if [ "${FORCE_CLEAN_CMAKE:-0}" = "1" ] || \
   { [ -d "${_cmake_build}" ] && ! grep -q '^install:' "${_cmake_build}/Makefile" 2>/dev/null; }; then
  echo "==> wiping incomplete cmake_build: ${_cmake_build}"
  rm -rf "${_cmake_build}"
fi

echo "==> build Milvus GPU/HIP (log: ${LOG})"
echo "    CMAKE_EXTRA_ARGS=${CMAKE_EXTRA_ARGS}"
echo "    MILVUS_HIP_INSTALL_PREFIX=${MILVUS_HIP_INSTALL_PREFIX}"
cd "${MILVUS_DIR}"

# Makefile target milvus-gpu → build-cpp-gpu → core_build.sh -g (MILVUS_GPU_VERSION=ON).
# CMAKE_EXTRA_ARGS is consumed by core_build.sh (-DMILVUS_KNOWHERE_SOURCE_DIR=...).
# Unset INSTALL_PREFIX so core_build.sh uses its default (internal/core/output) and does
# not inherit the hipVS prefix (which would break Milvus install layout).
# hipVS path is passed via MILVUS_HIP_INSTALL_PREFIX / ROCMDS_INSTALL_PREFIX only.
unset INSTALL_PREFIX
set +e
{
  echo "==== $(date -Is) Layer3 milvus build ===="
  echo "MILVUS_DIR=${MILVUS_DIR}"
  echo "MILVUS_HIP_INSTALL_PREFIX=${MILVUS_HIP_INSTALL_PREFIX}"
  echo "ROCM_PATH=${ROCM_PATH}"
  echo "CMAKE_EXTRA_ARGS=${CMAKE_EXTRA_ARGS}"
  export MILVUS_GPU_VERSION=ON
  if [ -f Makefile ] && grep -q '^milvus-gpu:' Makefile; then
    echo "==== make milvus-gpu ===="
    make milvus-gpu
  else
    bash scripts/core_build.sh -t Release -g
  fi
} 2>&1 | tee "${LOG}"
_build_rc=${PIPESTATUS[0]}
set -e

if [ "${_build_rc}" -ne 0 ]; then
  echo ""
  echo "BUILD FAILED (exit ${_build_rc}). CMake / make errors:" >&2
  grep -iE 'CMake Error|Configuring incomplete|No rule to make target|fatal error:|undefined reference|ConanException|Error downloading' \
    "${LOG}" | tail -50 >&2 || true
  echo "" >&2
  echo "Full log: ${LOG}" >&2
  echo "Tip: FORCE_CLEAN_CMAKE=1 SKIP_CLONE=1 bash scripts/build_milvus_layer3.sh" >&2
  exit "${_build_rc}"
fi

# Locate binary
_bin=""
for c in \
  "${MILVUS_DIR}/bin/milvus" \
  "${MILVUS_DIR}/cmake_build/milvus" \
  "${MILVUS_DIR}/internal/core/output/bin/milvus" \
  "${MILVUS_DIR}/out/bin/milvus"
do
  if [ -x "$c" ]; then _bin="$c"; break; fi
done

echo ""
echo "BUILD OK (or make reported success)"
if [ -n "${_bin}" ]; then
  echo "  milvus: ${_bin}"
else
  echo "  NOTE: milvus binary not found in common paths; search:"
  echo "    find ${MILVUS_DIR} -name milvus -type f 2>/dev/null | head"
fi
echo ""
echo "Next:"
echo "  bash ${REPO_ROOT}/scripts/run_milvus_gpu_smoke.sh"
