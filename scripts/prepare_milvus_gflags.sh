#!/usr/bin/env bash
# Diagnose / mitigate "flag 'help' was defined more than once" for HIP Milvus.
#
# Cause: Milvus and Knowhere Conan each statically embed a different gflags
# package_id (different options/namespace). Both register DEFINE_string(help,...).
#
# Usage:
#   bash scripts/prepare_milvus_gflags.sh
#   source <(bash scripts/prepare_milvus_gflags.sh --export)
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
MILVUS_DIR="${MILVUS_DIR:-${WORKDIR}/milvus}"
SHIM_DIR="${SHIM_DIR:-${WORKDIR}/libshims}"
EXPORT_ONLY=0
for a in "$@"; do
  case "$a" in
    --export) EXPORT_ONLY=1 ;;
  esac
done

_milvus_bin=""
for c in "${MILVUS_DIR}/bin/milvus" "${MILVUS_DIR}/internal/core/output/bin/milvus"; do
  if [ -x "$c" ]; then _milvus_bin="$c"; break; fi
done
if [ -z "${_milvus_bin}" ]; then
  echo "ERROR: milvus binary not found" >&2
  exit 1
fi

echo "==> milvus: ${_milvus_bin}"
echo "==> ldd gflags/glog:"
ldd "${_milvus_bin}" 2>/dev/null | grep -iE 'gflags|glog' || echo "  (none — likely static gflags inside binary/.so)"

echo "==> shared objects that contain FlagRegisterer:"
_found=0
while IFS= read -r -d '' f; do
  if nm -D "$f" 2>/dev/null | grep -q 'FlagRegisterer'; then
    echo "  $f"
    _found=1
  fi
done < <(find "${MILVUS_DIR}/internal/core/output/lib" "${MILVUS_DIR}/lib" \
  -name '*.so*' -print0 2>/dev/null || true)
# Also scan DT_NEEDED dirs
while IFS= read -r lib; do
  [ -z "$lib" ] && continue
  path=$(echo "$lib" | awk '/=>/ {print $3}')
  [ -z "$path" ] || [ "$path" = "not" ] && continue
  if nm -D "$path" 2>/dev/null | grep -q 'FlagRegisterer'; then
    echo "  $path"
    _found=1
  fi
done < <(ldd "${_milvus_bin}" 2>/dev/null | grep -iE 'gflags|glog' || true)

if [ "${_found}" -eq 0 ]; then
  echo "  (no dynamic FlagRegisterer — two *static* gflags copies are in the link)"
fi

# Prefer a single *threaded* shared libgflags for LD_PRELOAD (helps when one
# side is dynamic). Does not fix two static copies inside the same ELF.
_GFLAGS_SO=""
for c in \
  /usr/lib/x86_64-linux-gnu/libgflags.so.2.2 \
  /usr/lib/x86_64-linux-gnu/libgflags.so \
  "${SHIM_DIR}/libgflags.so"
do
  if [ -e "$c" ] && nm -D "$c" 2>/dev/null | grep -q FlagRegisterer; then
    _GFLAGS_SO="$c"
    break
  fi
done

mkdir -p "${SHIM_DIR}"
if [ -n "${_GFLAGS_SO}" ]; then
  ln -sfn "${_GFLAGS_SO}" "${SHIM_DIR}/libgflags_gflagsns.so"
  echo "==> shim link: ${SHIM_DIR}/libgflags_gflagsns.so -> ${_GFLAGS_SO}"
else
  echo "WARNING: no threaded libgflags.so with FlagRegisterer found" >&2
  echo "  sudo apt-get install -y libgflags2.2 libgflags-dev" >&2
fi

# Show Conan gflags package ids that appear in the binary's strings (from error).
echo "==> gflags build paths referenced in binary (if any):"
strings "${_milvus_bin}" 2>/dev/null | grep -E 'gflags/.*/src/gflags' | sort -u | head -20 || true

echo ""
echo "If you still see 'flag help was defined more than once', both copies are"
echo "static-linked. Then rebuild so Milvus+Knowhere share one Conan gflags id,"
echo "or link gflags shared-only. Quick retry with LD_PRELOAD:"
echo ""
echo "  unset LD_PRELOAD"
echo "  export LD_PRELOAD=${SHIM_DIR}/libgflags_gflagsns.so"
echo "  export LD_LIBRARY_PATH=${WORKDIR}/install/lib:/opt/rocm/lib:\$LD_LIBRARY_PATH"
echo "  ${_milvus_bin} run standalone"

if [ "${EXPORT_ONLY}" -eq 1 ]; then
  echo "export LD_PRELOAD=${SHIM_DIR}/libgflags_gflagsns.so"
  echo "export MILVUS_GFLAGS_SHIM=${SHIM_DIR}/libgflags_gflagsns.so"
fi
