#!/usr/bin/env bash
# Layer 1.5: check IVF gtest log for "testPacker mask mismatch" debug lines.
# Usage:
#   bash scripts/check_ivf_packer_mismatch.sh [logfile]
#   bash scripts/check_ivf_packer_mismatch.sh --run [--filter 'AnnIVFFlatTest/...']
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.0.2}"
LOG="${WORKDIR}/packer_debug_float.log"
GTEST_FILTER="${GTEST_FILTER:-AnnIVFFlatTest/AnnIVFFlatTestF_float.*}"
RUN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [logfile]
       $(basename "$0") --run [--filter 'GTEST_FILTER'] [--log PATH]

Check for debug lines: testPacker mask mismatch

Environment (optional):
  WORKDIR=${WORKDIR}
  INSTALL_PREFIX=${INSTALL_PREFIX}
  ROCM_PATH=${ROCM_PATH}

Exit codes:
  0  at least one mismatch line found (debug patch is active)
  1  log exists but no mismatch line (patch missing or all cases pass)
  2  log missing / gtest binary missing / --run failed
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --run) RUN=1; shift ;;
    --filter) GTEST_FILTER="${2:?--filter requires a value}"; shift 2 ;;
    --log) LOG="${2:?--log requires a path}"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) LOG="$1"; shift ;;
  esac
done

if [ "$RUN" -eq 1 ]; then
  GTEST="${WORKDIR}/hipVS/cpp/build/gtests/NEIGHBORS_ANN_IVF_FLAT_TEST"
  if [ ! -x "$GTEST" ]; then
    echo "ERROR: gtest not found: $GTEST" >&2
    echo "Rebuild hipVS with --limit-tests=NEIGHBORS_ANN_IVF_FLAT_TEST first." >&2
    exit 2
  fi
  export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
  echo "Running: $GTEST --gtest_filter='${GTEST_FILTER}'"
  echo "Log: $LOG"
  "$GTEST" --gtest_filter="${GTEST_FILTER}" 2>&1 | tee "$LOG"
  GTEST_RC="${PIPESTATUS[0]}"
  echo "gtest exit: ${GTEST_RC}"
fi

if [ ! -f "$LOG" ]; then
  echo "ERROR: log not found: $LOG" >&2
  exit 2
fi

PATTERN='testPacker mask mismatch'
COUNT="$(grep -c "$PATTERN" "$LOG" 2>/dev/null || true)"
LOG_LINES="$(wc -l < "$LOG" | tr -d ' ')"
FAILED_COUNT="$(grep -c '\[  FAILED  \]' "$LOG" 2>/dev/null || true)"
HIPVS_TEST="${WORKDIR}/hipVS/cpp/tests/neighbors/ann_ivf_flat.cuh"

echo "Log: $LOG"
echo "Log lines: ${LOG_LINES}"
echo "Pattern: $PATTERN"
echo "Count: ${COUNT}"
echo "Gtest [  FAILED  ] count: ${FAILED_COUNT}"

if [ "${COUNT}" -eq 0 ]; then
  echo "RESULT: NOT FOUND — no debug mismatch lines in log."
  if [ "${LOG_LINES}" -lt 5 ]; then
    echo "Hint: log looks empty or stale — run Step 4b:"
    echo "  bash scripts/check_ivf_packer_mismatch.sh --run"
  fi
  if [ -f "${HIPVS_TEST}" ]; then
    if grep -q 'testPacker mask mismatch' "${HIPVS_TEST}"; then
      echo "Hint: debug patch IS in source — rebuild + re-run gtest after patch."
    else
      echo "Hint: debug patch NOT in source — apply then clean-rebuild:"
      echo "  bash scripts/apply_hipvs_packer_debug_patch.sh ${WORKDIR}/hipVS"
    fi
  fi
  if [ "${FAILED_COUNT}" -gt 0 ]; then
    echo "Hint: gtests failed but no debug lines — binary built without debug patch."
    echo "  ./build.sh clean && rm -rf cpp/build  # then rebuild (see porting doc Step 4)"
  fi
  exit 1
fi

echo "RESULT: FOUND"
echo "--- first mismatch ---"
grep -m1 "$PATTERN" "$LOG"
echo "--- context (gtest case) ---"
grep -B8 -m1 "$PATTERN" "$LOG" | grep -E '\[  FAILED  \]|GetParam\(\)|testPacker mask mismatch' || true

if [ "${COUNT}" -gt 1 ]; then
  echo "--- note: ${COUNT} mismatch lines total; Step 5 starts from the first ---"
fi

exit 0
