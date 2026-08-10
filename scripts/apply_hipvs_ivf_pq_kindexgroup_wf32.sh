#!/usr/bin/env bash
# Force IVF-PQ kIndexGroupSize=32 under CUCO_USE_WARPSIZE_32 (gfx1100 / RDNA3).
#
# Hypothesis: raft::warp_size() is still 64 at compile time, so PQ list layout
# interleaves in groups of 64 while the device wave is 32 — hurts coalescing in
# compute_similarity_kernel. FLAT already has this fix (patch 0002); PQ did not.
#
# Usage:
#   bash scripts/apply_hipvs_ivf_pq_kindexgroup_wf32.sh [path/to/hipVS]
#
# Then rebuild hipVS (libcuvs + python) with USE_WARPSIZE_32=ON and rebuild the
# IVF_PQ index (layout is baked at build time).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPVS_ROOT="${1:-${WORKDIR:-$HOME/rocmds_check_gfx1100}/hipVS}"
F="$HIPVS_ROOT/cpp/include/cuvs/neighbors/ivf_pq.hpp"
PATCH="${REPO_ROOT}/patches/hipvs/0003-ivf-pq-kIndexGroupSize-warp32-gfx1100.patch"

if [[ ! -f "$F" ]]; then
  echo "ERROR: missing $F" >&2
  exit 1
fi

if grep -q 'CUCO_USE_WARPSIZE_32' "$F" && grep -q 'kIndexGroupSize = 32' "$F"; then
  echo "OK: IVF-PQ kIndexGroupSize=32 already present in $F"
  exit 0
fi

if [[ -d "$HIPVS_ROOT/.git" ]] && [[ -f "$PATCH" ]]; then
  if (cd "$HIPVS_ROOT" && git apply --check "$PATCH" 2>/dev/null); then
    (cd "$HIPVS_ROOT" && git apply "$PATCH")
    echo "Applied via git: $(basename "$PATCH")"
    exit 0
  fi
fi

python3 - "$F" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """/** Size of the interleaved group. */
constexpr static uint32_t kIndexGroupSize = raft::warp_size();
"""
new = """/** Size of the interleaved group. */
#if defined(CUCO_USE_WARPSIZE_32) && CUCO_USE_WARPSIZE_32
constexpr static uint32_t kIndexGroupSize = 32;
#else
constexpr static uint32_t kIndexGroupSize = raft::warp_size();
#endif
"""
if old not in text:
    # Already patched or layout changed
    if "kIndexGroupSize = 32" in text and "CUCO_USE_WARPSIZE_32" in text:
        print(f"OK: already patched {path}")
        raise SystemExit(0)
    print("ERROR: expected kIndexGroupSize = raft::warp_size() block not found", file=sys.stderr)
    for i, line in enumerate(text.splitlines(), 1):
        if "kIndexGroupSize" in line:
            print(f"  {i}: {line}", file=sys.stderr)
    raise SystemExit(2)
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print(f"Patched {path}")
PY

echo ""
echo "Next (AMD lab):"
echo "  cd \"\$HIPVS_ROOT\"   # $HIPVS_ROOT"
echo "  # rebuild with USE_WARPSIZE_32=ON (same as Layer 1 / FLAT packer)"
echo "  INSTALL_PREFIX=\$WORKDIR/install ./build.sh libcuvs python \\"
echo "    '--cmake-args=\"-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF\"' \\"
echo "    --gpu-arch=gfx1100"
echo "  # reinstall python/cuvs into ~/hipvs-bench-venv, then re-bench IVF_PQ"
echo "  INDEX_TYPE=IVF_PQ M=32 P99_SAMPLE=0 bash scripts/run_hipvs_ivf_bench.sh"
