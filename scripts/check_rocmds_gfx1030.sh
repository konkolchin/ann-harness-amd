#!/usr/bin/env bash
set -u

ARCH="gfx1030"
WORKDIR="${HOME}/rocmds_check"
LOGDIR="${WORKDIR}/logs"
SUMMARY="${WORKDIR}/SUMMARY.txt"

mkdir -p "$LOGDIR"
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

log "=== Environment check ==="
run_cmd rocminfo bash -lc 'rocminfo | grep -i -E "Name|gfx"' || true
run_cmd hipcc_version hipcc --version || true
run_cmd rocm_smi /opt/rocm/bin/rocm-smi || true

cd "$WORKDIR" || exit 1

export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake
export HIP_DIR=/opt/rocm/lib/cmake/hip
export rocthrust_DIR=/opt/rocm/lib/cmake/rocthrust
export RAPIDS_CMAKE_ROCM_ORG=ROCm
export RAPIDS_CMAKE_ROCM_DS_ORG=ROCm-DS

if [ ! -d hipRaft/.git ]; then
  run_cmd hipraft_clone git clone https://github.com/ROCm-DS/hipRaft.git hipRaft || true
fi
if [ ! -d hipVS/.git ]; then
  run_cmd hipvs_clone git clone https://github.com/ROCm-DS/hipVS.git hipVS || true
fi

run_cmd hipraft_build bash -lc "cd '$WORKDIR/hipRaft' && ./build.sh clean && ./build.sh libraft tests --gpu-arch='${ARCH}'"
HIPRAFT_RC=$?
run_cmd hipvs_build bash -lc "cd '$WORKDIR/hipVS' && ./build.sh clean && ./build.sh libcuvs tests --gpu-arch='${ARCH}'"
HIPVS_RC=$?

log "hipRAFT exit code: $HIPRAFT_RC"
log "hipVS exit code: $HIPVS_RC"
if [ "$HIPRAFT_RC" -eq 0 ]; then log "hipRAFT: PASS"; else log "hipRAFT: FAIL"; fi
if [ "$HIPVS_RC" -eq 0 ]; then log "hipVS: PASS"; else log "hipVS: FAIL"; fi
log "Summary: $SUMMARY"
