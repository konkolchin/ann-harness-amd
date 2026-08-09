#!/usr/bin/env bash
# Fix pickup_next_parents for gfx1100 (wf32):
# 1) Only first warp participates (guard was commented out upstream).
# 2) Rank with lane_id, not raw threadIdx.x (breaks when blockDim > 32).
# 3) Use uint32 ballot mask when warp_size()==32 (same class as move_invalid).
set -euo pipefail

HIPVS_ROOT="${1:-${WORKDIR:-$HOME/rocmds_check_gfx1100}/hipVS}"
F="$HIPVS_ROOT/cpp/src/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh"

if [[ ! -f "$F" ]]; then
  echo "ERROR: missing $F" >&2
  exit 1
fi

if grep -q 'pickup_next_parents: wf32 lane_id ballot fix' "$F"; then
  echo "OK: pickup_next_parents fix already present"
  exit 0
fi

python3 - "$F" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()

# Enforce first-warp-only + lane_id ranking + uint32 mask on wf32
old = re.compile(
    r"  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;\n"
    r"  // if \(threadIdx\.x >= 32\) return;\n"
    r"\n"
    r"  for \(std::uint32_t i = threadIdx\.x; i < search_width; i \+= raft::warp_size\(\)\) \{\n"
    r"    next_parent_indices\[i\] = utils::get_max_value<INDEX_T>\(\);\n"
    r"  \}\n"
    r"  std::uint32_t itopk_max = internal_topk_size;\n"
    r"  if \(itopk_max % raft::warp_size\(\)\) \{\n"
    r"    itopk_max \+= raft::warp_size\(\) - \(itopk_max % raft::warp_size\(\)\);\n"
    r"  \}\n"
    r"  std::uint32_t num_new_parents = 0;\n"
    r"  for \(std::uint32_t j = threadIdx\.x; j < itopk_max; j \+= raft::warp_size\(\)\) \{\n"
    r"    std::uint32_t jj = j;\n"
    r"    if \(TOPK_BY_BITONIC_SORT\) \{ jj = device::swizzling\(j\); \}\n"
    r"    INDEX_T index;\n"
    r"    int new_parent = 0;\n"
    r"    if \(j < internal_topk_size\) \{\n"
    r"      index = internal_topk_indices\[jj\];\n"
    r"      if \(\(index & index_msb_1_mask\) == 0\) \{  // check if most significant bit is set\n"
    r"        new_parent = 1;\n"
    r"      \}\n"
    r"    \}\n"
    r"    const bitmask_type ballot_mask = __ballot_sync\(__activemask\(\), new_parent\);\n"
    r"    if \(new_parent\) \{\n"
    r"      const auto i =\n"
    r"        raft::__POPC\(ballot_mask & \(\(static_cast<bitmask_type>\(1\) << threadIdx\.x\) - 1\)\) \+\n"
    r"        num_new_parents;\n"
    r"      if \(i < search_width\) \{\n"
    r"        next_parent_indices\[i\] = jj;\n"
    r"        // set most significant bit as used node\n"
    r"        internal_topk_indices\[jj\] \|= index_msb_1_mask;\n"
    r"      \}\n"
    r"    \}\n"
    r"    num_new_parents \+= raft::__POPC\(ballot_mask\);\n"
    r"    if \(num_new_parents >= search_width\) \{ break; \}\n"
    r"  \}\n"
    r"  if \(threadIdx\.x == 0 && \(num_new_parents == 0\)\) \{ \*terminate_flag = 1; \}\n"
    r"\}",
    re.M,
)

new = """  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  // pickup_next_parents: wf32 lane_id ballot fix
  // Only the first warp may run this; ballot ranking must use lane_id, and on
  // AMD wf32 bitmask_type is uint64_t so the prefix mask must be uint32_t.
  if (threadIdx.x >= raft::warp_size()) { return; }
  const unsigned lane_id = threadIdx.x;  // == threadIdx.x % warp_size() here

  for (std::uint32_t i = lane_id; i < search_width; i += raft::warp_size()) {
    next_parent_indices[i] = utils::get_max_value<INDEX_T>();
  }
  std::uint32_t itopk_max = internal_topk_size;
  if (itopk_max % raft::warp_size()) {
    itopk_max += raft::warp_size() - (itopk_max % raft::warp_size());
  }
  std::uint32_t num_new_parents = 0;
  using ballot_mask_t = std::conditional_t<raft::warp_size() == 32, uint32_t, bitmask_type>;
  for (std::uint32_t j = lane_id; j < itopk_max; j += raft::warp_size()) {
    std::uint32_t jj = j;
    if (TOPK_BY_BITONIC_SORT) { jj = device::swizzling(j); }
    INDEX_T index;
    int new_parent = 0;
    if (j < internal_topk_size) {
      index = internal_topk_indices[jj];
      if ((index & index_msb_1_mask) == 0) {  // check if most significant bit is set
        new_parent = 1;
      }
    }
    const ballot_mask_t ballot_mask =
      static_cast<ballot_mask_t>(__ballot_sync(__activemask(), new_parent));
    if (new_parent) {
      const auto i =
        raft::__POPC(ballot_mask & ((static_cast<ballot_mask_t>(1) << lane_id) - 1)) +
        num_new_parents;
      if (i < search_width) {
        next_parent_indices[i] = jj;
        // set most significant bit as used node
        internal_topk_indices[jj] |= index_msb_1_mask;
      }
    }
    num_new_parents += raft::__POPC(ballot_mask);
    if (num_new_parents >= search_width) { break; }
  }
  if (lane_id == 0 && (num_new_parents == 0)) { *terminate_flag = 1; }
}"""

m = old.search(text)
if not m:
    sys.exit(
        "ERROR: could not match pickup_next_parents body — whitespace drift?\n"
        "Inspect the function manually around the ballot_mask line."
    )
path.write_text(text[: m.start()] + new + text[m.end() :])
print(f"Patched {path}")
PY

if ! grep -q '#include <type_traits>' "$F"; then
  sed -i '0,/#pragma once/s//#pragma once\n\n#include <cstdint>\n#include <type_traits>/' "$F"
fi

echo "Verify:"
grep -n 'pickup_next_parents: wf32 lane_id ballot fix\|ballot_mask_t\|lane_id' "$F" | head -20
