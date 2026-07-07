#!/usr/bin/env bash
# Apply Layer 2 Knowhere patches (HIP CMake + GPU device shim).
#
# Usage:
#   bash scripts/apply_knowhere_layer2_patches.sh [path/to/knowhere]
#   bash scripts/apply_knowhere_layer2_patches.sh --no-reset [path/to/knowhere]
#
# By default the script hard-resets knowhere to branch 2.5 and removes untracked
# files left by prior patch attempts (git checkout -- . is NOT enough).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/patches/knowhere"
DO_RESET=1
KNOWHERE_DIR=""

for arg in "$@"; do
  case "$arg" in
    --no-reset) DO_RESET=0 ;;
    --reset) DO_RESET=1 ;;
    -*) echo "Unknown option: $arg" >&2; exit 1 ;;
    *) KNOWHERE_DIR="$arg" ;;
  esac
done
KNOWHERE_DIR="${KNOWHERE_DIR:-${HOME}/rocmds_check_gfx1100/knowhere}"

apply_patch() {
  local patch="$1"
  local name
  name="$(basename "$patch")"

  if git apply --check "$patch" 2>/dev/null; then
    git apply "$patch"
    echo "Applied: ${name}"
    return 0
  fi

  if git apply --reverse --check "$patch" 2>/dev/null; then
    echo "Already applied: ${name}"
    return 0
  fi

  echo "ERROR: cannot apply ${name}." >&2
  echo "  Re-run without --no-reset to hard-reset knowhere to branch 2.5." >&2
  exit 1
}

verify_patches() {
  local ok=1
  if ! grep -q 'libhipcuvs_preproject.cmake' CMakeLists.txt; then
    echo "VERIFY FAIL: CMakeLists.txt missing libhipcuvs_preproject include" >&2
    ok=0
  fi
  if ! grep -q 'Early WITH_HIP before project' CMakeLists.txt; then
    echo "VERIFY FAIL: CMakeLists.txt missing patch 0010 early WITH_HIP block" >&2
    ok=0
  fi
  if ! grep -q 'knowhere_cuvs_hip OBJECT' cmake/libs/knowhere_hip_host_fixup.cmake; then
    echo "VERIFY FAIL: missing patch 0027 knowhere_cuvs_hip OBJECT library" >&2
    ok=0
  fi
  for f in cmake/libs/libhipcuvs.cmake \
           cmake/libs/libhipcuvs_preproject.cmake \
           cmake/libs/knowhere_hip_host_fixup.cmake \
           cmake/libs/knowhere_hip_link.cmake; do
    if [ ! -f "$f" ]; then
      echo "VERIFY FAIL: missing $f" >&2
      ok=0
    fi
  done
  if [ "$ok" -eq 0 ]; then
    exit 1
  fi
  echo "VERIFY OK: all Layer 2 patch markers present"
}

if [ ! -d "${KNOWHERE_DIR}/.git" ]; then
  echo "knowhere git checkout not found: ${KNOWHERE_DIR}" >&2
  echo "  git clone -b 2.5 --depth 1 https://github.com/zilliztech/knowhere.git ${KNOWHERE_DIR}" >&2
  exit 1
fi

cd "${KNOWHERE_DIR}"

if [ "$DO_RESET" -eq 1 ]; then
  echo "Resetting ${KNOWHERE_DIR} to clean upstream knowhere branch 2.5..."
  git fetch origin 2>/dev/null || true
  if git show-ref --verify --quiet refs/heads/2.5; then
    git checkout 2.5
  elif git show-ref --verify --quiet refs/remotes/origin/2.5; then
    git checkout -B 2.5 origin/2.5
  else
    echo "ERROR: branch 2.5 not found in ${KNOWHERE_DIR}" >&2
    exit 1
  fi
  if git show-ref --verify --quiet refs/remotes/origin/2.5; then
    git reset --hard origin/2.5
  else
    git reset --hard HEAD
  fi
  # Remove untracked patch artifacts (libhipcuvs.cmake etc.) that break git apply.
  git clean -fd
  echo "Reset complete: $(git log -1 --oneline)"
fi

shopt -s nullglob
patches=( "${PATCH_DIR}"/[0-9]*.patch )
IFS=$'\n' patches=( $(printf '%s\n' "${patches[@]}" | sort) )
unset IFS
for patch in "${patches[@]}"; do
  apply_patch "$patch"
done

verify_patches

echo ""
echo "Layer 2 patches applied under ${KNOWHERE_DIR}"
echo "Next:"
echo "  bash ${REPO_ROOT}/scripts/configure_knowhere_hip.sh"
echo "  # or manually: export INSTALL_PREFIX=\$HOME/rocmds_check_gfx1100/install"
echo "  # cmake .. -DWITH_CUVS=ON -DWITH_HIP=ON -DCMAKE_PREFIX_PATH=\"\$INSTALL_PREFIX;/opt/rocm\" ..."
