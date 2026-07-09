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
# gflags: Conan glog needs gflags::FlagRegisterer (_ZN6gflags14...).
# Apt libgflags has google:: (_ZN6google14...) — wrong namespace.
# Conan gflags is often static-only (libgflags.a); wrap it into a shim .so.
# ---------------------------------------------------------------------------
_gflags_preload=""

# 1) Prefer an existing shared lib that exports gflags::FlagRegisterer.
while IFS= read -r _cand; do
  [ -n "${_cand}" ] || continue
  if nm -D "${_cand}" 2>/dev/null | grep -q '_ZN6gflags14FlagRegisterer'; then
    _gflags_preload="${_cand}"
    break
  fi
done < <(find "${HOME}/.conan" "${HOME}/.conan2" -type f \( -name 'libgflags.so' -o -name 'libgflags.so.*' \) ! -name '*nothreads*' 2>/dev/null | head -40)

# 2) Else build a shared shim from Conan static archive (gflags:: symbols).
#    ConanCenter often packages only libgflags_nothreads.a (still gflags:: ns).
if [ -z "${_gflags_preload}" ]; then
  _gflags_a=""
  while IFS= read -r _cand; do
    [ -n "${_cand}" ] || continue
    if nm "${_cand}" 2>/dev/null | grep -q '_ZN6gflags14FlagRegisterer'; then
      _gflags_a="${_cand}"
      break
    fi
  done < <(find "${HOME}/.conan" "${HOME}/.conan2" -type f \( \
      -name 'libgflags.a' -o -name 'libgflags_nothreads.a' \) 2>/dev/null | head -40)

  if [ -z "${_gflags_a}" ]; then
    echo "No Conan libgflags*.a with gflags:: symbols; forcing Conan rebuild..."
    conan install gflags/2.2.2@ --build=gflags -s build_type=Release || true
    while IFS= read -r _cand; do
      [ -n "${_cand}" ] || continue
      if nm "${_cand}" 2>/dev/null | grep -q '_ZN6gflags14FlagRegisterer'; then
        _gflags_a="${_cand}"
        break
      fi
    done < <(find "${HOME}/.conan" "${HOME}/.conan2" -type f \( \
        -name 'libgflags.a' -o -name 'libgflags_nothreads.a' \) 2>/dev/null | head -40)
  fi

  if [ -z "${_gflags_preload}" ] && [ -n "${_gflags_a}" ]; then
    mkdir -p "${SHIM_DIR}"
    _gflags_preload="${SHIM_DIR}/libgflags_gflagsns.so"
    echo "Building gflags:: shim from ${_gflags_a} -> ${_gflags_preload}"
    g++ -shared -fPIC -o "${_gflags_preload}" \
      -Wl,--whole-archive "${_gflags_a}" -Wl,--no-whole-archive \
      -lpthread -lrt
    if ! nm -D "${_gflags_preload}" | grep -q '_ZN6gflags14FlagRegisterer'; then
      echo "ERROR: shim built but still missing gflags::FlagRegisterer" >&2
      exit 1
    fi
  fi
  unset _gflags_a
fi

if [ -z "${_gflags_preload}" ]; then
  echo "ERROR: cannot provide gflags::FlagRegisterer for Conan glog." >&2
  echo "  Inspect Conan gflags package:" >&2
  echo "    ls -la ~/.conan/data/gflags/2.2.2/_/_/package/*/lib/" >&2
  echo "    nm ~/.conan/data/gflags/2.2.2/_/_/package/*/lib/libgflags_nothreads.a | grep FlagRegisterer | head" >&2
  exit 1
fi

prepend_libdir "$(dirname "${_gflags_preload}")"
export LD_PRELOAD="${_gflags_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
echo "Using gflags preload: ${_gflags_preload}"
unset _cand _gflags_preload

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
