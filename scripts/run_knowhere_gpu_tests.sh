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

prepend_libdir() {
  local d="$1"
  [ -n "$d" ] && [ -d "$d" ] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${d}:"*) ;;
    *) export LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

# Conan run env first when available.
_conan_gen="${BUILD_DIR}/Release/generators"
if [ -f "${_conan_gen}/conanrun.sh" ]; then
  # shellcheck disable=SC1091
  source "${_conan_gen}/conanrun.sh"
elif [ -f "${_conan_gen}/activate_run.sh" ]; then
  # shellcheck disable=SC1091
  source "${_conan_gen}/activate_run.sh"
fi
unset _conan_gen

# Conan glog needs the *threaded* libgflags.so FlagRegisterer. System
# libgflags_nothreads.so.2.2 does NOT provide it — put Conan gflags first.
_gflags_so=""
for _cand in \
  "${HOME}/.conan/data/gflags"/*/package/*/lib/libgflags.so \
  "${HOME}/.conan/data/gflags"/*/package/*/lib/libgflags.so.* \
  "${HOME}/.conan2/p"/*/p/lib/libgflags.so; do
  if [ -e "${_cand}" ]; then
    _gflags_so="${_cand}"
    break
  fi
done
# Broader search if glob missed (Conan package hash paths vary).
if [ -z "${_gflags_so}" ]; then
  _gflags_so="$(find "${HOME}/.conan" "${HOME}/.conan2" -type f -name 'libgflags.so' 2>/dev/null | head -1 || true)"
fi
if [ -n "${_gflags_so}" ]; then
  prepend_libdir "$(dirname "${_gflags_so}")"
  echo "Using Conan gflags: ${_gflags_so}"
else
  echo "WARNING: Conan libgflags.so not found under ~/.conan — glog may fail against system nothreads" >&2
  echo "  Try: cd ${BUILD_DIR} && conan install .. --build=missing -s build_type=Release -o with_cuvs=True -o with_ut=True" >&2
fi
unset _cand _gflags_so

# Also ensure the Conan glog package dir is visible (same install as knowhere_tests).
_glog_so="$(find "${HOME}/.conan" "${HOME}/.conan2" -type f -name 'libglog.so.1' 2>/dev/null | head -1 || true)"
if [ -n "${_glog_so}" ]; then
  prepend_libdir "$(dirname "${_glog_so}")"
fi
unset _glog_so

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
# install/rocm before /usr so hipRAFT libs win; Conan gflags already prepended above.
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}:/usr/lib/x86_64-linux-gnu"

# Never LD_PRELOAD apt spdlog (ABI mismatch with embedded Knowhere symbols).
unset LD_PRELOAD

_libknowhere="${BUILD_DIR}/libknowhere.so"
if nm -D "${_libknowhere}" 2>/dev/null | grep -qE ' [TW] _ZN6spdlog.*set_pattern'; then
  echo "OK: spdlog set_pattern embedded in libknowhere.so"
else
  echo "WARNING: spdlog set_pattern not found as T/W in libknowhere.so (rebuild with patch 0047)" >&2
fi
unset _libknowhere

echo "spdlog DT_NEEDED:"
ldd "${BUILD_DIR}/libknowhere.so" | grep spdlog || true
echo "gflags/glog resolution:"
ldd "${TEST_BIN}" 2>/dev/null | grep -E 'gflags|glog' || true
# Must NOT be libgflags_nothreads for Conan glog.
if ldd "${TEST_BIN}" 2>/dev/null | grep -q 'libgflags_nothreads'; then
  if ! ldd "${TEST_BIN}" 2>/dev/null | grep -qE 'libgflags\.so'; then
    echo "ERROR: only libgflags_nothreads is loaded; Conan glog needs libgflags.so" >&2
    echo "  find ~/.conan -name 'libgflags.so' | head" >&2
    exit 1
  fi
fi
echo ""

if [ "$#" -eq 0 ]; then
  set -- 'Test Gpu Index Search L2 Metric'
fi

exec "${TEST_BIN}" "$@"
