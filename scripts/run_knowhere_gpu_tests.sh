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

# Conan run env when available.
_conan_gen="${BUILD_DIR}/Release/generators"
if [ -f "${_conan_gen}/conanrun.sh" ]; then
  # shellcheck disable=SC1091
  source "${_conan_gen}/conanrun.sh"
elif [ -f "${_conan_gen}/activate_run.sh" ]; then
  # shellcheck disable=SC1091
  source "${_conan_gen}/activate_run.sh"
fi
unset _conan_gen

# Conan glog needs gflags::FlagRegisterer from the *threaded* libgflags.
# System libgflags_nothreads.so does NOT export it. Conan often installs only
# static libgflags.a — then we must use /usr/lib/.../libgflags.so (apt).
_gflags_shared=""
for _cand in \
  /usr/lib/x86_64-linux-gnu/libgflags.so.2.2 \
  /usr/lib/x86_64-linux-gnu/libgflags.so.2 \
  /usr/lib/x86_64-linux-gnu/libgflags.so \
  /usr/lib/libgflags.so; do
  if [ -e "${_cand}" ]; then
    _gflags_shared="${_cand}"
    break
  fi
done
if [ -z "${_gflags_shared}" ]; then
  _gflags_shared="$(find "${HOME}/.conan" "${HOME}/.conan2" -type f \( -name 'libgflags.so' -o -name 'libgflags.so.*' \) ! -name '*nothreads*' 2>/dev/null | head -1 || true)"
fi

if [ -n "${_gflags_shared}" ]; then
  prepend_libdir "$(dirname "${_gflags_shared}")"
  # Force this SONAME ahead of nothreads (glog may DT_NEEDED gflags without path).
  export LD_PRELOAD="${_gflags_shared}${LD_PRELOAD:+:${LD_PRELOAD}}"
  echo "Using threaded gflags: ${_gflags_shared}"
  if ! nm -D "${_gflags_shared}" 2>/dev/null | grep -q FlagRegisterer; then
    echo "WARNING: ${_gflags_shared} has no FlagRegisterer symbol" >&2
  fi
else
  echo "ERROR: threaded libgflags.so not found." >&2
  echo "  Install: sudo apt-get install -y libgflags2.2 libgflags-dev" >&2
  echo "  Or rebuild Conan gflags shared: conan install gflags/2.2.2@ --build=missing -o gflags:shared=True" >&2
  echo "  Conan gflags package libs:" >&2
  find "${HOME}/.conan/data/gflags" -path '*/package/*/lib/*' 2>/dev/null | head -20 >&2 || true
  exit 1
fi
unset _cand _gflags_shared

_glog_so="$(find "${HOME}/.conan" "${HOME}/.conan2" -type f -name 'libglog.so.1' 2>/dev/null | head -1 || true)"
if [ -n "${_glog_so}" ]; then
  prepend_libdir "$(dirname "${_glog_so}")"
fi
unset _glog_so

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}:/usr/lib/x86_64-linux-gnu"

_libknowhere="${BUILD_DIR}/libknowhere.so"
if nm -D "${_libknowhere}" 2>/dev/null | grep -qE ' [TW] .*spdlog.*set_pattern'; then
  echo "OK: spdlog set_pattern embedded in libknowhere.so"
else
  echo "NOTE: nm did not list spdlog set_pattern as T/W (may still be OK if HEADER_ONLY weak)" >&2
fi
unset _libknowhere

echo "spdlog DT_NEEDED:"
ldd "${BUILD_DIR}/libknowhere.so" | grep spdlog || true
echo "gflags/glog resolution (with LD_PRELOAD=${LD_PRELOAD:-}):"
ldd "${TEST_BIN}" 2>/dev/null | grep -E 'gflags|glog' || true
echo ""

if [ "$#" -eq 0 ]; then
  set -- 'Test Gpu Index Search L2 Metric'
fi

exec "${TEST_BIN}" "$@"
