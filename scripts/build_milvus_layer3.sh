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

export INSTALL_PREFIX ROCM_PATH HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"
export PATH="${ROCM_PATH}/llvm/bin:${HOME}/.local/bin:${PATH}"
export CMAKE_PREFIX_PATH="${INSTALL_PREFIX};${ROCM_PATH};${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

if [ ! -f "${INSTALL_PREFIX}/lib/cmake/cuvs/cuvs-config.cmake" ]; then
  echo "ERROR: hipVS not found at ${INSTALL_PREFIX}/lib/cmake/cuvs/" >&2
  echo "  Complete Layer 1 / 1.5 first." >&2
  exit 1
fi

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
