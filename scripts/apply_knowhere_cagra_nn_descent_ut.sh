#!/usr/bin/env bash
# Apply patches/knowhere/0052 (CAGRA NN_DESCENT in Catch2) and rebuild knowhere_tests.
#
# Usage on amd-rx7900xtx:
#   export WORKDIR=~/rocmds_check_gfx1100
#   bash ~/ann-harness-amd/scripts/apply_knowhere_cagra_nn_descent_ut.sh
#   RUN_CATCH2=1 bash ~/ann-harness-amd/scripts/apply_knowhere_cagra_nn_descent_ut.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
KH="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
PATCH="${REPO_ROOT}/patches/knowhere/0052-cagra-default-nn-descent-in-gpu-ut.patch"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [ ! -d "${KH}" ]; then
  echo "ERROR: ${KH} missing" >&2
  exit 1
fi
if [ ! -f "${PATCH}" ]; then
  echo "ERROR: ${PATCH} missing — git pull ann-harness-amd" >&2
  exit 1
fi

cd "${KH}"
# Idempotent: skip if already present
if grep -q 'BUILD_ALGO.*=.*"NN_DESCENT"' tests/ut/test_gpu_search.cc 2>/dev/null \
  || grep -q 'BUILD_ALGO.*NN_DESCENT' tests/ut/test_gpu_search.cc 2>/dev/null; then
  echo "==> NN_DESCENT already in cagra_gen (or nearby); skipping patch apply"
else
  echo "==> applying ${PATCH}"
  if git apply --check "${PATCH}" 2>/dev/null; then
    git apply "${PATCH}"
  else
    # Fuzzy: insert after ITOPK_SIZE line in cagra_gen
    echo "==> git apply failed; trying line insert"
    python3 - <<'PY'
from pathlib import Path
p = Path("tests/ut/test_gpu_search.cc")
text = p.read_text(encoding="utf-8")
needle = "json[knowhere::indexparam::ITOPK_SIZE] = 128;"
insert = (
    needle
    + '\n        json[knowhere::indexparam::BUILD_ALGO] = "NN_DESCENT";'
)
if 'BUILD_ALGO] = "NN_DESCENT"' in text:
    print("already patched")
elif needle not in text:
    raise SystemExit(f"cannot find ITOPK_SIZE line in {p}")
else:
    # only first occurrence inside cagra_gen
    text = text.replace(needle, insert, 1)
    p.write_text(text, encoding="utf-8")
    print("inserted BUILD_ALGO NN_DESCENT after ITOPK_SIZE")
PY
  fi
fi

grep -n 'BUILD_ALGO\|ITOPK_SIZE\|cagra_gen' tests/ut/test_gpu_search.cc | head -40

echo "==> rebuild knowhere_tests (incremental)"
export ROCM_PATH
export PATH="${ROCM_PATH}/llvm/bin:${PATH}"
# Prefer existing build dir
if [ -d build ] && [ -f build/build.ninja -o -f build/Makefile ]; then
  cmake --build build -j"$(nproc)" --target knowhere_tests 2>&1 | tail -40
else
  echo "ERROR: no build/ — rebuild Knowhere WITH_HIP first (Layer-2)" >&2
  exit 1
fi

KT=""
for c in build/tests/ut/knowhere_tests build/knowhere_tests; do
  if [ -x "$c" ]; then KT="$(pwd)/$c"; break; fi
done
echo "knowhere_tests=${KT}"

if [ "${RUN_CATCH2:-0}" = "1" ]; then
  export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
  export LD_LIBRARY_PATH="${KH}/build:${WORKDIR}/install/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
  LOG="${WORKDIR}/logs/cagra_catch2_nn_descent_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "${WORKDIR}/logs"
  echo "==> Catch2 -> ${LOG}"
  set +e
  "${KT}" "Test All GPU Index" -s 2>&1 | tee "${LOG}"
  echo "exit=$?"
  set -e
  grep -n -A5 'FAILED:' "${LOG}" || echo "(no FAILED lines)"
fi
