#!/usr/bin/env bash
# Backport AMD-Ecosystem/hipVS 26.03 fix: uint32 ballot mask when warp_size==32
# in CAGRA move_invalid_to_end_of_list (filtered search path).
set -euo pipefail

HIPVS_ROOT="${1:-${WORKDIR:-$HOME/rocmds_check_gfx1100}/hipVS}"
F="$HIPVS_ROOT/cpp/src/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh"

if [[ ! -f "$F" ]]; then
  echo "ERROR: missing $F" >&2
  exit 1
fi

if grep -q 'intentionally not using bitmask_type' "$F"; then
  echo "OK: patch already present in $F"
  exit 0
fi

if ! grep -q 'const bitmask_type who_has_invalid = raft::ballot' "$F"; then
  echo "ERROR: expected buggy ballot line not found — file layout changed?" >&2
  grep -n 'who_has_invalid\|move_invalid_to_end' "$F" | head -20 >&2 || true
  exit 2
fi

python3 - "$F" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
pat = re.compile(
    r"        // Check if the index is invalid\n"
    r"        const auto I_found_invalid\s*=\s*\(index == invalid_index\);\n"
    r"        const bitmask_type who_has_invalid = raft::ballot\(I_found_invalid, __activemask\(\)\);\n"
    r"        // if a value that is loaded by a smaller lane id thread, shift the array\n"
    r"        if \(\(who_has_invalid << \(raft::warp_size\(\) - lane_id\)\) and i > 0\) \{\n"
    r"          index_array\[i - 1\]\s*=\s*index;\n"
    r"          distance_array\[i - 1\]\s*=\s*distance;\n"
    r"        \}\n"
    r"\n"
    r"        found_invalid = who_has_invalid;",
    re.M,
)
new = """        // Check if the index is invalid
        const auto I_found_invalid = (index == invalid_index);
        using mask_type = std::conditional_t<raft::warp_size() == 32, uint32_t, uint64_t>;
        // We're intentionally not using bitmask_type(uint64_t) here.
        // Note the expression `(who_has_invalid << (raft::warp_size() - lane_id)` which is trying
        // to compute the following: It is trying to check if there is any smaller lane that has
        // found invalid index. By left shifting the bitmask by (warp_size - lane_id), All the lanes
        // that found an invalid index will have it's bit shifted out except for the lanes with
        // smaller lane_id than the current lane. For this to work correctly, the mask_type should
        // not be wider than the actual warp size.
        const mask_type who_has_invalid = raft::ballot(I_found_invalid, __activemask());
        // if a value that is loaded by a smaller lane id thread, shift the array
        if ((who_has_invalid << (raft::warp_size() - lane_id)) and i > 0) {
          index_array[i - 1]    = index;
          distance_array[i - 1] = distance;
        }

        found_invalid = who_has_invalid;"""

m = pat.search(text)
if not m:
    sys.exit("ERROR: could not match buggy block (whitespace drift?)")
path.write_text(text[: m.start()] + new + text[m.end() :])
print(f"Patched {path}")
PY

if ! grep -q '#include <type_traits>' "$F"; then
  sed -i '0,/#pragma once/s//#pragma once\n\n#include <cstdint>\n#include <type_traits>/' "$F"
fi

echo "Verify:"
grep -n 'intentionally not using bitmask_type\|mask_type who_has_invalid' "$F" | head
