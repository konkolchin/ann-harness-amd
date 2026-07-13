#!/usr/bin/env bash
# Apply Layer 3 Milvus patches (HIP Knowhere FetchContent redirect).
#
# Usage:
#   bash scripts/apply_milvus_layer3_patches.sh [path/to/milvus]
#   bash scripts/apply_milvus_layer3_patches.sh --no-reset [path/to/milvus]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/patches/milvus"
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

# Prefer python3, then python (single interpreter — no broken double-heredoc).
_PY="$(command -v python3 || true)"
if [ -z "${_PY}" ]; then
  _PY="$(command -v python || true)"
fi

normalize_patch() {
  local src="$1"
  local dst="$2"
  if [ -n "${_PY}" ]; then
    "${_PY}" - "${src}" "${dst}" <<'PY'
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8", errors="replace")
lines = text.splitlines(True)
out = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("diff --git "):
        parts = line.strip().split()
        a = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else (parts[2] if len(parts) > 2 else "")
        b = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") else (parts[3] if len(parts) > 3 else a)
        out.append(line if line.endswith("\n") else line + "\n")
        i += 1
        saw_paths = False
        while i < len(lines) and not lines[i].startswith("diff --git "):
            cur = lines[i]
            if cur.startswith("--- ") or cur.startswith("+++ "):
                saw_paths = True
                out.append(cur if cur.endswith("\n") else cur + "\n")
                i += 1
                continue
            if cur.startswith("@@") and not saw_paths and a:
                out.append(f"--- a/{a}\n")
                out.append(f"+++ b/{b}\n")
                saw_paths = True
                out.append(cur if cur.endswith("\n") else cur + "\n")
                i += 1
                continue
            out.append(cur if cur.endswith("\n") else cur + "\n")
            i += 1
        continue
    out.append(line if line.endswith("\n") else line + "\n")
    i += 1
dst.write_text("".join(out), encoding="utf-8", newline="\n")
PY
  else
    # Fallback: strip CR only (patches already ship with ---/+++).
    tr -d '\r' <"${src}" >"${dst}"
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
  echo "  git clone -b v2.5.4 --depth 1 https://github.com/milvus-io/milvus.git ${MILVUS_DIR}" >&2
  exit 1
fi

cd "${MILVUS_DIR}"

if [ "$DO_RESET" -eq 1 ]; then
  echo "Resetting ${MILVUS_DIR} to clean v2.5.4..."
  git fetch origin tag v2.5.4 --no-tags 2>/dev/null || git fetch origin 2>/dev/null || true
  git reset --hard HEAD
  git clean -fd
  if git show-ref --verify --quiet refs/tags/v2.5.4; then
    git checkout -B v2.5.4 v2.5.4
    git reset --hard v2.5.4
  elif git show-ref --verify --quiet refs/remotes/origin/v2.5.4; then
    git checkout -B v2.5.4 origin/v2.5.4
    git reset --hard origin/v2.5.4
  else
    # Shallow clone of tag may be detached at the tag commit — stay there.
    echo "NOTE: no v2.5.4 branch/tag ref; using current HEAD=$(git log -1 --oneline)" >&2
    git checkout -B v2.5.4 HEAD
  fi
  git clean -fd
  echo "Reset complete: $(git log -1 --oneline)"
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
echo "VERIFY OK: Layer 3 Milvus patches present"
echo "Next: bash ${REPO_ROOT}/scripts/build_milvus_layer3.sh"
