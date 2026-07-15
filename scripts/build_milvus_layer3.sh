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

# Knowhere libfaiss.cmake requires CONFIG-mode xxHash. Prefer Layer-2 Conan
# generators; otherwise install xxHash into INSTALL_PREFIX (already on prefix path).
KNOWHERE_CONAN_GENERATORS="${KNOWHERE_CONAN_GENERATORS:-${KNOWHERE_DIR}/build/Release/generators}"
if [ -f "${KNOWHERE_CONAN_GENERATORS}/xxHashConfig.cmake" ] || \
   [ -f "${KNOWHERE_CONAN_GENERATORS}/xxhash-config.cmake" ]; then
  echo "==> xxHash from Knowhere Conan: ${KNOWHERE_CONAN_GENERATORS}"
  export KNOWHERE_CONAN_GENERATORS
elif [ -f "${INSTALL_PREFIX}/lib/cmake/xxHash/xxHashConfig.cmake" ]; then
  echo "==> xxHash already in INSTALL_PREFIX"
else
  _xx=$(find "${KNOWHERE_DIR}/build" "${HOME}/.conan/data" -name 'xxHashConfig.cmake' 2>/dev/null | head -1 || true)
  if [ -n "${_xx}" ]; then
    KNOWHERE_CONAN_GENERATORS="$(cd "$(dirname "${_xx}")" && pwd)"
    echo "==> xxHash found: ${KNOWHERE_CONAN_GENERATORS}"
    export KNOWHERE_CONAN_GENERATORS
  else
    echo "==> installing xxHash into ${INSTALL_PREFIX} (Knowhere FAISS needs CONFIG package)"
    _xx_src="${WORKDIR}/src/xxHash"
    mkdir -p "${WORKDIR}/src"
    if [ ! -d "${_xx_src}/.git" ]; then
      git clone --depth 1 --branch v0.8.2 https://github.com/Cyan4973/xxHash.git "${_xx_src}"
    fi
    cmake -S "${_xx_src}/cmake_unofficial" -B "${_xx_src}/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DBUILD_SHARED_LIBS=ON
    cmake --build "${_xx_src}/build" -j"$(nproc)"
    cmake --install "${_xx_src}/build"
    if [ ! -f "${INSTALL_PREFIX}/lib/cmake/xxHash/xxHashConfig.cmake" ]; then
      echo "ERROR: xxHashConfig.cmake missing after install under ${INSTALL_PREFIX}" >&2
      exit 1
    fi
  fi
fi
export CMAKE_PREFIX_PATH="${KNOWHERE_CONAN_GENERATORS:-};${INSTALL_PREFIX};${ROCM_PATH};${CMAKE_PREFIX_PATH:-}"
# Drop leading empty segment if generators unset
export CMAKE_PREFIX_PATH="$(echo "${CMAKE_PREFIX_PATH}" | sed 's/^;//')"

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
# Force CONFIG-mode xxHash even when INSTALL_PREFIX is clobbered by core_build.sh.
_xx_cfg=""
for _c in \
  "${KNOWHERE_CONAN_GENERATORS:-}/xxHashConfig.cmake" \
  "${MILVUS_HIP_INSTALL_PREFIX}/lib/cmake/xxHash/xxHashConfig.cmake" \
  "${MILVUS_HIP_INSTALL_PREFIX}/lib64/cmake/xxHash/xxHashConfig.cmake"
do
  if [ -f "${_c}" ]; then _xx_cfg="${_c}"; break; fi
done
if [ -n "${_xx_cfg}" ]; then
  CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DxxHash_DIR=$(dirname "${_xx_cfg}")"
  echo "==> CMAKE xxHash_DIR=$(dirname "${_xx_cfg}")"
else
  echo "ERROR: xxHashConfig.cmake not found; cannot configure Knowhere FAISS" >&2
  exit 1
fi
export CMAKE_EXTRA_ARGS

echo "==> install Milvus build deps (idempotent)"
if [ -x "${MILVUS_DIR}/scripts/install_deps.sh" ]; then
  bash "${MILVUS_DIR}/scripts/install_deps.sh" || true
fi

echo "==> build Milvus GPU/HIP (log: ${LOG})"
echo "    CMAKE_EXTRA_ARGS=${CMAKE_EXTRA_ARGS}"
cd "${MILVUS_DIR}"

# Makefile target milvus-gpu → build-cpp-gpu → core_build.sh -g (MILVUS_GPU_VERSION=ON).
# CMAKE_EXTRA_ARGS is consumed by core_build.sh (-DMILVUS_KNOWHERE_SOURCE_DIR=...).
set +e
{
  echo "==== $(date -Is) Layer3 milvus build ===="
  echo "MILVUS_DIR=${MILVUS_DIR}"
  echo "INSTALL_PREFIX=${INSTALL_PREFIX}"
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
  echo "BUILD FAILED (exit ${_build_rc}). Last errors:" >&2
  grep -iE 'error:|fatal error:|FAILED:|undefined reference' "${LOG}" | tail -40 >&2 || tail -50 "${LOG}" >&2
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
