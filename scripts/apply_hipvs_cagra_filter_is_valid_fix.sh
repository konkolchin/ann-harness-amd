#!/usr/bin/env bash
# Fix post-filter WarpScan validity check in single-CTA CAGRA.
# Bug: compares (idx & ~msb) == invalid_index, which is never true for
# invalid=0xFFFFFFFF (MSB cleared → 0x7FFFFFFF). Invalids are treated as
# valid during compaction. Use equality to full invalid_index instead.
set -euo pipefail

HIPVS_ROOT="${1:-${WORKDIR:-$HOME/rocmds_check_gfx1100}/hipVS}"
F="$HIPVS_ROOT/cpp/src/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh"

if [[ ! -f "$F" ]]; then
  echo "ERROR: missing $F" >&2
  exit 1
fi

if grep -q 'is_valid_index: compare full invalid_index' "$F"; then
  echo "OK: is_valid_index fix already present"
  exit 0
fi

python3 - "$F" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = """        const std::uint32_t is_valid_index =
          (result_indices_buffer[src_position] & (~index_msb_1_mask)) == invalid_index ? 0 : 1;"""
new = """        // is_valid_index: compare full invalid_index (not masked).
        // (idx & ~msb) == 0xFFFFFFFF is impossible when msb was cleared.
        const std::uint32_t is_valid_index =
          (result_indices_buffer[src_position] == invalid_index) ? 0 : 1;"""
if old not in text:
    # tolerate spacing
    import re
    pat = re.compile(
        r"        const std::uint32_t is_valid_index =\n"
        r"          \(result_indices_buffer\[src_position\] & \(~index_msb_1_mask\)\) == invalid_index \? 0 : 1;"
    )
    m = pat.search(text)
    if not m:
        sys.exit("ERROR: could not find is_valid_index assignment")
    text = text[: m.start()] + new + text[m.end() :]
else:
    text = text.replace(old, new, 1)
path.write_text(text)
print(f"Patched {path}")
PY

grep -n 'is_valid_index: compare full invalid_index\|is_valid_index' "$F" | head -10
