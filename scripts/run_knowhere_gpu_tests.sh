#!/usr/bin/env bash
# Run Knowhere GPU unit tests (Layer 2).
#
# Usage:
#   bash scripts/run_knowhere_gpu_tests.sh
#   bash scripts/run_knowhere_gpu_tests.sh 'Test Gpu Index Search L2 Metric'
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
KNOWHERE_DIR="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
BUILD_DIR="${BUILD_DIR:-${KNOWHERE_DIR}/build}"
TEST_BIN="${TEST_BIN:-${BUILD_DIR}/tests/ut/knowhere_tests}"

if [ ! -x "${TEST_BIN}" ]; then
  echo "ERROR: knowhere_tests not found: ${TEST_BIN}" >&2
  echo "  Build first: bash scripts/build_knowhere_layer2.sh" >&2
  exit 1
fi

# Restore quarantined hipRAFT spdlog if present (libcuvs may DT_NEEDED SONAME 1.14).
shopt -s nullglob
for _f in "${INSTALL_PREFIX}/lib"/libspdlog.so*.hipraft-bak; do
  _orig="${_f%.hipraft-bak}"
  if [ ! -e "${_orig}" ]; then
    mv -v "${_f}" "${_orig}"
  fi
done
unset _f _orig
shopt -u nullglob

# Conan run env (glog needs gflags, etc.). Prefer generated conanrun scripts.
_conan_gen="${BUILD_DIR}/Release/generators"
if [ -f "${_conan_gen}/conanrun.sh" ]; then
  # shellcheck disable=SC1091
  source "${_conan_gen}/conanrun.sh"
elif [ -f "${_conan_gen}/activate_run.sh" ]; then
  # shellcheck disable=SC1091
  source "${_conan_gen}/activate_run.sh"
else
  # Fallback: add Conan package lib dirs that knowhere_tests / libglog need.
  while IFS= read -r _libdir; do
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${_libdir}:"*) ;;
      *) export LD_LIBRARY_PATH="${_libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
  done < <(
    find "${HOME}/.conan" "${HOME}/.conan2" -type f \( -name 'libgflags.so*' -o -name 'libglog.so*' \) \
      2>/dev/null | sed 's|/[^/]*$||' | sort -u | head -40
  )
  unset _libdir
fi
unset _conan_gen

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Do NOT LD_PRELOAD apt spdlog: Knowhere embeds set_pattern (W) with old ABI (ERKSs);
# apt spdlog is cxx11 ABI and breaks resolution / can confuse the loader.
unset LD_PRELOAD

_libknowhere="${BUILD_DIR}/libknowhere.so"
if nm -D "${_libknowhere}" 2>/dev/null | grep -qE ' [TW] .*spdlog.*set_pattern'; then
  echo "OK: spdlog set_pattern embedded in libknowhere.so"
else
  echo "WARNING: spdlog set_pattern not found as T/W in libknowhere.so (rebuild with patch 0047)" >&2
fi
unset _libknowhere

echo "spdlog DT_NEEDED:"
ldd "${BUILD_DIR}/libknowhere.so" | grep spdlog || true
echo "gflags check:"
ldd "${TEST_BIN}" 2>/dev/null | grep -E 'gflags|glog' || true
echo ""

if [ "$#" -eq 0 ]; then
  set -- 'Test Gpu Index Search L2 Metric'
fi

exec "${TEST_BIN}" "$@"
