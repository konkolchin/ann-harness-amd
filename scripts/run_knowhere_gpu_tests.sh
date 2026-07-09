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
SHIM_DIR="${SHIM_DIR:-${WORKDIR}/libshims}"

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

# ---------------------------------------------------------------------------
# gflags shim for Conan glog:
#   Conan glog needs  gflags::FlagRegisterer  (_ZN6gflags14...)
#   Conan/apt gflags often export google::     (_ZN6google14...)
#   "google" and "gflags" are both length 6 → safe mangled rename via objcopy.
# ---------------------------------------------------------------------------
mkdir -p "${SHIM_DIR}"
_gflags_preload="${SHIM_DIR}/libgflags_gflagsns.so"

_need_rebuild=1
if [ -e "${_gflags_preload}" ] && nm -D "${_gflags_preload}" 2>/dev/null | grep -q '_ZN6gflags14FlagRegisterer'; then
  _need_rebuild=0
  echo "Reusing gflags shim: ${_gflags_preload}"
fi

if [ "${_need_rebuild}" -eq 1 ]; then
  _gflags_a=""

  # Dead-simple discovery: same path the user already verified with nm.
  # (Complex find/glob + missing ~/.conan2 was failing under set -e/pipefail.)
  set +e
  _gflags_a="$(ls -1 \
    "${HOME}"/.conan/data/gflags/*/_/_/package/*/lib/libgflags_nothreads.a \
    "${HOME}"/.conan/data/gflags/*/_/_/package/*/lib/libgflags.a \
    2>/dev/null | head -1)"
  set -e

  if [ -z "${_gflags_a}" ] || [ ! -f "${_gflags_a}" ]; then
    echo "ERROR: no Conan libgflags*.a under ~/.conan/data/gflags/" >&2
    echo "  Tried:" >&2
    ls -la "${HOME}"/.conan/data/gflags/*/_/_/package/*/lib/ 2>&1 | head -40 >&2 || true
    exit 1
  fi
  echo "Found Conan gflags archive: ${_gflags_a}"
  # Do not use `nm | grep -q` under pipefail: grep -q exits early → nm SIGPIPE → false negative.
  if ! nm "${_gflags_a}" 2>/dev/null | grep FlagRegisterer | head -1 | grep -q .; then
    echo "ERROR: ${_gflags_a} has no FlagRegisterer symbols" >&2
    nm "${_gflags_a}" 2>/dev/null | grep -i flag | head -5 >&2 || true
    exit 1
  fi

  echo "Building gflags shim from ${_gflags_a}"
  _tmp_so="${SHIM_DIR}/libgflags_shim_tmp.so"
  g++ -shared -fPIC -o "${_tmp_so}" \
    -Wl,--whole-archive "${_gflags_a}" -Wl,--no-whole-archive \
    -lpthread -lrt

  if nm -D "${_tmp_so}" | grep -q '_ZN6gflags14FlagRegisterer'; then
    mv -f "${_tmp_so}" "${_gflags_preload}"
    echo "Shim already has gflags:: namespace"
  elif nm -D "${_tmp_so}" | grep -q '_ZN6google14FlagRegisterer'; then
    # Rename google:: → gflags:: in mangled names (same identifier length).
    _map="${SHIM_DIR}/gflags_redefine.map"
    nm -D "${_tmp_so}" | awk '{print $NF}' | grep '6google' | sort -u | while read -r _sym; do
      printf '%s %s\n' "${_sym}" "${_sym//6google/6gflags}"
    done > "${_map}"
    echo "Renaming google:: → gflags:: ($(wc -l < "${_map}") symbols)"
    objcopy --redefine-syms="${_map}" "${_tmp_so}" "${_gflags_preload}"
    rm -f "${_tmp_so}" "${_map}"
  else
    echo "ERROR: FlagRegisterer not found in either namespace after linking shim" >&2
    nm -D "${_tmp_so}" | grep -i FlagRegisterer | head >&2 || true
    exit 1
  fi

  if ! nm -D "${_gflags_preload}" | grep -q '_ZN6gflags14FlagRegisterer'; then
    echo "ERROR: shim missing gflags::FlagRegisterer after rename" >&2
    exit 1
  fi
  echo "Built ${_gflags_preload}"
fi
unset _need_rebuild _gflags_a _cand _tmp_so _map _sym

prepend_libdir "$(dirname "${_gflags_preload}")"
export LD_PRELOAD="${_gflags_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
echo "Using gflags preload: ${_gflags_preload}"
unset _gflags_preload

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
echo "gflags/glog resolution (LD_PRELOAD=${LD_PRELOAD:-}):"
ldd "${TEST_BIN}" 2>/dev/null | grep -E 'gflags|glog' || true
echo ""

if [ "$#" -eq 0 ]; then
  set -- 'Test Gpu Index Search L2 Metric'
fi

exec "${TEST_BIN}" "$@"
