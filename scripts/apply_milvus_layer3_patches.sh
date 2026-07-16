#!/usr/bin/env bash
# Apply Layer 3 Milvus patches (HIP Knowhere FetchContent redirect).
#
# Usage:
#   bash scripts/apply_milvus_layer3_patches.sh [path/to/milvus]
#   bash scripts/apply_milvus_layer3_patches.sh --no-reset [path/to/milvus]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/patches/milvus"
NORM_PY="${REPO_ROOT}/scripts/_normalize_milvus_patch.py"
DO_RESET=1
MILVUS_DIR=""

for arg in "$@"; do
  case "$arg" in
    --no-reset) DO_RESET=0 ;;
    --reset) DO_RESET=1 ;;
    -*) echo "Unknown option: $arg" >&2; exit 1 ;;
    *) MILVUS_DIR="$arg" ;;
  esac
done
MILVUS_DIR="${MILVUS_DIR:-${HOME}/rocmds_check_gfx1100/milvus}"

_PY="$(command -v python3 || true)"
if [ -z "${_PY}" ]; then
  _PY="$(command -v python || true)"
fi
if [ -n "${_PY}" ] && ! "${_PY}" -c 'import sys' >/dev/null 2>&1; then
  _PY=""
fi

normalize_patch() {
  local src="$1"
  local dst="$2"
  # Prefer external helper (no bash heredoc — avoids truncated-script EOF bugs).
  if [ -n "${_PY}" ] && [ -f "${NORM_PY}" ]; then
    "${_PY}" "${NORM_PY}" "${src}" "${dst}"
  else
    # Fallback: strip CR only (patches already ship with ---/+++).
    tr -d $'\r' <"${src}" >"${dst}"
  fi
}

apply_patch() {
  local patch="$1"
  local name
  name="$(basename "$patch")"
  local norm
  norm="$(mktemp)"
  normalize_patch "$patch" "$norm"

  if git apply --check "$norm" 2>/dev/null; then
    git apply "$norm"
    echo "Applied: ${name}"
    rm -f "$norm"
    return 0
  fi
  if git apply --reverse --check "$norm" 2>/dev/null; then
    echo "Already applied: ${name}"
    rm -f "$norm"
    return 0
  fi
  if patch -p1 --dry-run -s <"$norm" >/dev/null 2>&1; then
    patch -p1 -s <"$norm"
    echo "Applied(patch): ${name}"
    rm -f "$norm"
    return 0
  fi
  echo "ERROR: cannot apply ${name}" >&2
  git apply --check "$norm" 2>&1 | tail -20 || true
  rm -f "$norm"
  exit 1
}

if [ ! -d "${MILVUS_DIR}/.git" ]; then
  echo "milvus git checkout not found: ${MILVUS_DIR}" >&2
  echo "  git clone --depth 1 --branch v2.5.4 https://github.com/milvus-io/milvus.git ${MILVUS_DIR}" >&2
  exit 1
fi

cd "${MILVUS_DIR}"

if [ "$DO_RESET" -eq 1 ]; then
  # Unambiguous local branch: a branch named "v2.5.4" collides with the tag.
  _MILVUS_LOCAL_BRANCH="${MILVUS_LOCAL_BRANCH:-milvus-v2.5.4}"
  echo "Resetting ${MILVUS_DIR} to clean refs/tags/v2.5.4 (branch ${_MILVUS_LOCAL_BRANCH})..."
  git fetch origin tag v2.5.4 --no-tags 2>/dev/null || git fetch origin 2>/dev/null || true
  git reset --hard HEAD
  git clean -fd
  if git show-ref --verify --quiet refs/tags/v2.5.4; then
    git checkout -B "${_MILVUS_LOCAL_BRANCH}" refs/tags/v2.5.4
    git reset --hard refs/tags/v2.5.4
  elif git show-ref --verify --quiet refs/remotes/origin/v2.5.4; then
    git checkout -B "${_MILVUS_LOCAL_BRANCH}" refs/remotes/origin/v2.5.4
    git reset --hard refs/remotes/origin/v2.5.4
  else
    echo "NOTE: no v2.5.4 tag/remote ref; using current HEAD=$(git log -1 --oneline)" >&2
    git checkout -B "${_MILVUS_LOCAL_BRANCH}" HEAD
  fi
  git clean -fd
  echo "Reset complete: $(git log -1 --oneline) (on ${_MILVUS_LOCAL_BRANCH})"
  unset _MILVUS_LOCAL_BRANCH
fi

shopt -s nullglob
patches=("${PATCH_DIR}"/[0-9]*.patch)
IFS=$'\n' patches=($(printf '%s\n' "${patches[@]}" | sort))
unset IFS
if [ "${#patches[@]}" -eq 0 ]; then
  echo "ERROR: no patches in ${PATCH_DIR}" >&2
  exit 1
fi
for p in "${patches[@]}"; do
  apply_patch "$p"
done

if ! grep -q 'MILVUS Layer3 HIP' internal/core/thirdparty/knowhere/CMakeLists.txt; then
  echo "VERIFY FAIL: Layer 3 HIP Knowhere markers missing" >&2
  exit 1
fi
if ! grep -q 'WITH_HIP ON' internal/core/thirdparty/knowhere/CMakeLists.txt; then
  echo "VERIFY FAIL: WITH_HIP not forced in knowhere CMakeLists" >&2
  exit 1
fi
if ! grep -q 'without CUDA language' internal/core/src/CMakeLists.txt; then
  echo "VERIFY FAIL: CUDA language skip patch missing in src/CMakeLists.txt" >&2
  exit 1
fi
if ! grep -q 'IsAdditionalScalarSupported(' internal/core/src/index/VectorDiskIndex.cpp \
  || ! grep -q 'is_partition_key_isolation' internal/core/src/index/VectorDiskIndex.cpp; then
  echo "VERIFY FAIL: IsAdditionalScalarSupported(bool) compat patch missing" >&2
  exit 1
fi
if grep -q 'wrapper_->template add_multi_data' internal/core/src/index/InvertedIndexTantivy.cpp; then
  echo "VERIFY FAIL: Tantivy clang template-kw patch not applied" >&2
  exit 1
fi
echo "VERIFY OK: Layer 3 Milvus patches present"
echo "Next: bash ${REPO_ROOT}/scripts/build_milvus_layer3.sh"
