#!/usr/bin/env bash
# Layer 1 on RX 7900 XTX (gfx1100): build hipRAFT + hipVS and run a small test subset.
# Run inside tmux on amd-rx7900xtx. Expect 1–3+ hours first time (downloads + compile).
set -u

ARCH="${ARCH:-gfx1100}"
BRANCH="${BRANCH:-release/rocmds-25.10}"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
LOGDIR="${WORKDIR}/logs"
SUMMARY="${WORKDIR}/SUMMARY.txt"

mkdir -p "$LOGDIR" "$INSTALL_PREFIX"
: > "$SUMMARY"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

run_cmd() {
  local name="$1"; shift
  local logfile="${LOGDIR}/${name}.log"
  echo "### CMD: $*" > "$logfile"
  "$@" >> "$logfile" 2>&1
  local rc=$?
  echo "### EXIT_CODE: $rc" >> "$logfile"
  return $rc
}

log "=== Layer 1 environment (${ARCH}, branch ${BRANCH}) ==="
run_cmd rocminfo bash -lc '/opt/rocm/bin/rocminfo | grep -E "Name:|Marketing|gfx"' || true
run_cmd hipcc_version bash -lc '/opt/rocm/bin/hipcc --version' || true
run_cmd rocm_smi /opt/rocm/bin/rocm-smi || true
run_cmd cmake_version cmake --version || true
run_cmd ninja_version ninja --version || true

export PATH="${HOME}/.local/bin:/opt/rocm/bin:${PATH}"

# Resolve ROCm prefix (/opt/rocm is usually a symlink to /opt/rocm-7.0.2 or /opt/rocm-6.4.4)
if [ -z "${ROCM_PATH:-}" ]; then
  for candidate in /opt/rocm /opt/rocm-7.0.2 /opt/rocm-6.4.4; do
    if [ -d "${candidate}/lib/cmake/rocthrust" ]; then
      ROCM_PATH="${candidate}"
      break
    fi
  done
  ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
fi
export ROCM_PATH
export CMAKE_PREFIX_PATH="${ROCM_PATH}:${ROCM_PATH}/lib/cmake:${INSTALL_PREFIX}/lib/cmake"
export HIP_DIR="${ROCM_PATH}/lib/cmake/hip"
export rocthrust_DIR="${ROCM_PATH}/lib/cmake/rocthrust"
export rocprim_DIR="${ROCM_PATH}/lib/cmake/rocprim"
export hipcub_DIR="${ROCM_PATH}/lib/cmake/hipcub"
export RAPIDS_CMAKE_ROCM_ORG=ROCm
export RAPIDS_CMAKE_ROCM_DS_ORG=ROCm-DS
export RAPIDS_CMAKE_ROCTHRUST_USE_LOCAL=ON
export PARALLEL_LEVEL="${PARALLEL_LEVEL:-$(nproc)}"
export INSTALL_PREFIX

# build.sh cmake-args regex needs literal double-quotes in argv: '--cmake-args="-D..."'
# (writing --cmake-args="-D..." lets bash strip them → Invalid option)

cd "$WORKDIR" || exit 1

patch_hipraft_rocthrust_version() {
  local vf="${WORKDIR}/hipRaft/versions.json"
  [ -f "$vf" ] || return 0
  # ROCm 7 ships rocthrust 4.0.0 — keep versions.json at 4.0.0.
  # ROCm 6.4 apt ships 3.3.x; patch only then or find_package fails and CPM hits
  # the sentinel git tag We-always-use-find_package-for-rocthrust.
  case "${ROCM_PATH}" in
    /opt/rocm-6.*|/opt/rocm/6.*) ;;
    *)
      if [[ "${ROCM_PATH}" == /opt/rocm ]] && [ -d /opt/rocm-6.4.4/lib/cmake/rocthrust ] \
        && [ ! -d /opt/rocm-7.0.2/lib/cmake/rocthrust ]; then
        : # /opt/rocm -> 6.4.x
      else
        log "ROCm 7+ detected (${ROCM_PATH}): leaving hipRaft rocthrust at 4.0.0"
        return 0
      fi
      ;;
  esac
  python3 <<'PY'
import json
from pathlib import Path
p = Path("'"$vf"'")
data = json.loads(p.read_text())
pkg = data.setdefault("packages", {}).setdefault("rocthrust", {})
pkg["version"] = "3.3.0"
p.write_text(json.dumps(data, indent=2) + "\n")
print("Patched", p, "-> rocthrust version 3.3.0 for ROCm 6.4.x")
PY
}

if [ ! -d hipRaft/.git ]; then
  run_cmd hipraft_clone git clone -b "$BRANCH" --depth 1 https://github.com/ROCm-DS/hipRaft.git hipRaft || true
fi
patch_hipraft_rocthrust_version
if [ ! -d hipVS/.git ]; then
  run_cmd hipvs_clone git clone -b "$BRANCH" --depth 1 https://github.com/ROCm-DS/hipVS.git hipVS || true
fi

log "=== Build hipRAFT (lib + tests) ==="
run_cmd hipraft_build bash -lc "
  cd '${WORKDIR}/hipRaft' &&
  ./build.sh clean &&
  rm -rf cpp/build _deps &&
  INSTALL_PREFIX='${INSTALL_PREFIX}' CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}' \
    rocthrust_DIR='${rocthrust_DIR}' rocprim_DIR='${rocprim_DIR}' hipcub_DIR='${hipcub_DIR}' \
    RAPIDS_CMAKE_ROCTHRUST_USE_LOCAL=ON \
    RAPIDS_CMAKE_ROCM_ORG=ROCm \
    ./build.sh libraft tests --compile-lib --cache-tool=ccache \
    '--cmake-args=\"-DUSE_WARPSIZE_32=ON\"' --gpu-arch='${ARCH}'
"
HIPRAFT_RC=$?

log "=== Build hipVS (libcuvs + tests, subset) ==="
run_cmd hipvs_build bash -lc "
  cd '${WORKDIR}/hipVS' &&
  ./build.sh clean &&
  INSTALL_PREFIX='${INSTALL_PREFIX}' CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}' \
    rocthrust_DIR='${rocthrust_DIR}' rocprim_DIR='${rocprim_DIR}' hipcub_DIR='${hipcub_DIR}' \
    RAPIDS_CMAKE_ROCTHRUST_USE_LOCAL=ON \
    RAPIDS_CMAKE_ROCM_ORG=ROCm \
    ./build.sh libcuvs tests --cache-tool=ccache \
    '--cmake-args=\"-DUSE_WARPSIZE_32=ON\"' --gpu-arch='${ARCH}' \
    --limit-tests='NEIGHBORS_ANN_IVF_FLAT_TEST;BRUTEFORCE_C_TEST'
"
HIPVS_RC=$?

log "=== Run small hipVS gtests (if built) ==="
GTEST_DIR="${WORKDIR}/hipVS/cpp/build/gtests"
if [ -x "${GTEST_DIR}/BRUTEFORCE_C_TEST" ]; then
  run_cmd bruteforce_test env LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}" \
    "${GTEST_DIR}/BRUTEFORCE_C_TEST" || true
fi
if [ -x "${GTEST_DIR}/NEIGHBORS_ANN_IVF_FLAT_TEST" ]; then
  run_cmd ivf_flat_test env LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}" \
    "${GTEST_DIR}/NEIGHBORS_ANN_IVF_FLAT_TEST" || true
fi

log "hipRAFT build exit code: ${HIPRAFT_RC}"
log "hipVS build exit code: ${HIPVS_RC}"
if [ "$HIPRAFT_RC" -eq 0 ]; then log "hipRAFT: PASS"; else log "hipRAFT: FAIL (see ${LOGDIR}/hipraft_build.log)"; fi
if [ "$HIPVS_RC" -eq 0 ]; then log "hipVS: PASS"; else log "hipVS: FAIL (see ${LOGDIR}/hipvs_build.log)"; fi
log "Install prefix: ${INSTALL_PREFIX}"
log "Summary file: ${SUMMARY}"
log "Logs: ${LOGDIR}/"
