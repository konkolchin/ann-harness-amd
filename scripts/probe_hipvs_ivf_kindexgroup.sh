#!/usr/bin/env bash
# Show IVF_FLAT / IVF_PQ kIndexGroupSize definitions in a hipVS tree.
# Use before/after the wave32 patches to confirm PQ still uses raft::warp_size().
set -euo pipefail

HIPVS_ROOT="${1:-${WORKDIR:-$HOME/rocmds_check_gfx1100}/hipVS}"
FLAT="$HIPVS_ROOT/cpp/include/cuvs/neighbors/ivf_flat.hpp"
PQ="$HIPVS_ROOT/cpp/include/cuvs/neighbors/ivf_pq.hpp"

for f in "$FLAT" "$PQ"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f" >&2
    exit 1
  fi
done

echo "==> $FLAT"
grep -n -A6 'kIndexGroupSize' "$FLAT" | head -40
echo ""
echo "==> $PQ"
grep -n -A6 'kIndexGroupSize' "$PQ" | head -40
echo ""

# Host-visible warpSize from the runtime (device prop). Compile-time
# raft::warp_size() may still be 64 even when this prints 32.
python3 - <<'PY' 2>/dev/null || true
try:
    import cupy as cp
    p = cp.cuda.runtime.getDeviceProperties(0)
    name = p.get("name", b"?")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    print(f"==> device prop warpSize={p.get('warpSize') or p.get('warp_size')} name={name!r}")
except Exception as e:
    print(f"==> cupy device prop unavailable: {e}")
PY

echo ""
echo "Expect after patches 0002+0003 (with CUCO_USE_WARPSIZE_32):"
echo "  both headers: kIndexGroupSize = 32 inside #if CUCO_USE_WARPSIZE_32"
echo "If PQ still shows only raft::warp_size() — apply:"
echo "  bash scripts/apply_hipvs_ivf_pq_kindexgroup_wf32.sh"
