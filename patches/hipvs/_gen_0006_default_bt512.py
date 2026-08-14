#!/usr/bin/env python3
"""Generate 0006-ivf-pq-default-blockdim-512-amd.patch (hardcode AMD prefer 512)."""
from __future__ import annotations

import difflib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "_tmp_sim.cuh"
DST = HERE / "_tmp_sim_bt512.cuh"
PATCH = HERE / "0006-ivf-pq-default-blockdim-512-amd.patch"
STOCK_URL = (
    "https://raw.githubusercontent.com/konkolchin/hipVS/"
    "release/rocmds-25.10/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh"
)

ANCHOR = """    occupancy_t<OutT, LutT, IvfSampleFilterT> cur(smem_size, n_threads, kernel, dev_props);
    if (cur.blocks_per_sm <= 0) {
      // For some reason, we still cannot make this kernel run. Skip the candidate.
      continue;
    }

    {
      // Try to reduce the number of threads to increase occupancy and data locality
      auto n_threads_tmp = n_threads_min;
"""

INSERT = """    occupancy_t<OutT, LutT, IvfSampleFilterT> cur(smem_size, n_threads, kernel, dev_props);
    if (cur.blocks_per_sm <= 0) {
      // For some reason, we still cannot make this kernel run. Skip the candidate.
      continue;
    }

    // Cap before auto-shrink (used by AMD prefer-512 below).
    const uint32_t n_threads_cap = n_threads;

    {
      // Try to reduce the number of threads to increase occupancy and data locality
      auto n_threads_tmp = n_threads_min;
"""

CLOSE_OLD = """          n_threads_tmp /= 2;
        }
      }
    }

    {
      if (selected_perf.occupancy <= 0.0  // no candidate yet
"""

CLOSE_NEW = """          n_threads_tmp /= 2;
        }
      }
    }

#if defined(__HIP_PLATFORM_AMD__)
    {
      // DXC gfx1100 PoC: prefer blockDim=512 for IVF-PQ compute_similarity.
      // Measured +28–36% search QPS vs stock auto-shrink on RX 7900 XTX
      // (ann-harness results/lib_bench/LAUNCH_KNOBS.md, 2026-08-14); recall flat.
      // Env HIPVS_IVF_PQ_BLOCK_THREADS (patch 0005) still overrides when set.
      const char* env_bt = std::getenv("HIPVS_IVF_PQ_BLOCK_THREADS");
      const bool env_overrides = (env_bt != nullptr && env_bt[0] != '\\0');
      constexpr uint32_t kAmdIvfPqPreferThreads = 512u;
      if (!env_overrides && kAmdIvfPqPreferThreads >= n_threads_gty &&
          kAmdIvfPqPreferThreads <= n_threads_cap) {
        auto smem_pref = smem_size_f(kAmdIvfPqPreferThreads);
        cudaError_t st_pref =
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pref);
        if (st_pref == cudaSuccess) {
          occupancy_t<OutT, LutT, IvfSampleFilterT> pref(
            smem_pref, kAmdIvfPqPreferThreads, kernel, dev_props);
          if (pref.blocks_per_sm > 0) {
            n_threads = kAmdIvfPqPreferThreads;
            smem_size = smem_pref;
            cur       = pref;
          }
        } else {
          RAFT_EXPECTS(st_pref == cudaGetLastError(),
                       "Tried to reset the expected cuda error code, but it didn't match the expectation");
        }
      }
    }
#endif

    {
      if (selected_perf.occupancy <= 0.0  // no candidate yet
"""


def main() -> None:
    if not SRC.is_file():
        print(f"fetching {STOCK_URL}")
        urllib.request.urlretrieve(STOCK_URL, SRC)
    t = SRC.read_text(encoding="utf-8")

    # getenv needs cstdlib if not present (0005 may already add it)
    if "#include <cstdlib>" not in t:
        old_inc = '#pragma once\n\n#include "../ivf_common.cuh"'
        new_inc = (
            "#pragma once\n\n"
            "#include <cstdlib>  // getenv (AMD IVF-PQ blockDim prefer)\n\n"
            '#include "../ivf_common.cuh"'
        )
        if old_inc not in t:
            raise SystemExit("include anchor missing")
        t = t.replace(old_inc, new_inc, 1)

    if ANCHOR not in t:
        raise SystemExit("occupancy anchor missing (is 0005 already applied?)")
    t = t.replace(ANCHOR, INSERT, 1)

    if CLOSE_OLD not in t:
        raise SystemExit("close anchor missing")
    t = t.replace(CLOSE_OLD, CLOSE_NEW, 1)

    DST.write_text(t, encoding="utf-8")
    a = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
    # Re-read stock for diff base — if we mutated includes on copy of stock, use original stock
    # SRC may equal original; for includes we mutated `t` from SRC content at start.
    # Rebuild `a` from freshly fetched stock without includes if we added includes to t only.
    stock = urllib.request.urlopen(STOCK_URL).read().decode("utf-8")
    a = stock.splitlines(keepends=True)
    b = t.splitlines(keepends=True)
    PATCH.write_text(
        "".join(
            difflib.unified_diff(
                a,
                b,
                fromfile="a/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh",
                tofile="b/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh",
                n=3,
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {PATCH.name}")
    print("prefer-512 marker", "kAmdIvfPqPreferThreads" in t)


if __name__ == "__main__":
    main()
