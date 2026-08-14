#!/usr/bin/env python3
"""Generate 0005-ivf-pq-force-launch-knobs-gfx1100.patch from stock hipVS header."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "_tmp_sim.cuh"
DST = HERE / "_tmp_sim_patched.cuh"
PATCH = HERE / "0005-ivf-pq-force-launch-knobs-gfx1100.patch"
STOCK_URL = (
    "https://raw.githubusercontent.com/ROCm-DS/hipVS/"
    "release/rocmds-25.10/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh"
)


def main() -> None:
    if not SRC.is_file():
        import urllib.request

        print(f"fetching {STOCK_URL}")
        urllib.request.urlretrieve(STOCK_URL, SRC)
    t = SRC.read_text(encoding="utf-8")

    old_inc = '#pragma once\n\n#include "../ivf_common.cuh"'
    new_inc = (
        "#pragma once\n\n"
        "#include <cstdlib>  // getenv (ann-harness launch knobs)\n"
        "#include <cstring>\n"
        "#include <cstdio>\n\n"
        '#include "../ivf_common.cuh"'
    )
    if old_inc not in t:
        raise SystemExit("include anchor missing")
    t = t.replace(old_inc, new_inc, 1)

    anchor = (
        "auto compute_similarity_select(const cudaDeviceProp& dev_props,\n"
        "                               bool manage_local_topk,\n"
        "                               int locality_hint,\n"
        "                               double preferred_shmem_carveout,\n"
        "                               uint32_t pq_bits,\n"
        "                               uint32_t pq_dim,\n"
        "                               uint32_t precomp_data_count,\n"
        "                               uint32_t n_queries,\n"
        "                               uint32_t n_probes,\n"
        "                               uint32_t topk) -> selected<OutT, LutT, IvfSampleFilterT>\n"
        "{\n"
        "  // Shared memory for storing the lookup table\n"
    )
    insert = (
        "auto compute_similarity_select(const cudaDeviceProp& dev_props,\n"
        "                               bool manage_local_topk,\n"
        "                               int locality_hint,\n"
        "                               double preferred_shmem_carveout,\n"
        "                               uint32_t pq_bits,\n"
        "                               uint32_t pq_dim,\n"
        "                               uint32_t precomp_data_count,\n"
        "                               uint32_t n_queries,\n"
        "                               uint32_t n_probes,\n"
        "                               uint32_t topk) -> selected<OutT, LutT, IvfSampleFilterT>\n"
        "{\n"
        "  // ann-harness gfx1100: launch A/B knobs (host-side, no math change)\n"
        "  //   HIPVS_IVF_PQ_FORCE_VARIANT=fast|no_basediff|no_smem_lut\n"
        "  //   HIPVS_IVF_PQ_FORCE_SMEM_LUT=0|1  + HIPVS_IVF_PQ_FORCE_PRECOMP=0|1\n"
        "  //   HIPVS_IVF_PQ_BLOCK_THREADS=128|256|512|1024\n"
        "  //   HIPVS_IVF_PQ_FORCE_NO_LOCAL_TOPK=1\n"
        "  //   HIPVS_IVF_PQ_PREFERRED_CARVEOUT=0.0..1.0\n"
        '  if (const char* e = std::getenv("HIPVS_IVF_PQ_FORCE_NO_LOCAL_TOPK");'
        " e != nullptr && e[0] == '1') {\n"
        "    manage_local_topk = false;\n"
        "  }\n"
        '  if (const char* e = std::getenv("HIPVS_IVF_PQ_PREFERRED_CARVEOUT");'
        " e != nullptr && e[0] != '\\0') {\n"
        "    preferred_shmem_carveout = std::atof(e);\n"
        "  }\n"
        "  int force_variant_idx = -1;  // 0=fast, 1=no_basediff, 2=no_smem_lut\n"
        '  if (const char* e = std::getenv("HIPVS_IVF_PQ_FORCE_VARIANT");'
        " e != nullptr && e[0] != '\\0') {\n"
        '    if (std::strcmp(e, "fast") == 0) {\n'
        "      force_variant_idx = 0;\n"
        '    } else if (std::strcmp(e, "no_basediff") == 0) {\n'
        "      force_variant_idx = 1;\n"
        '    } else if (std::strcmp(e, "no_smem_lut") == 0) {\n'
        "      force_variant_idx = 2;\n"
        "    } else {\n"
        '      RAFT_FAIL("HIPVS_IVF_PQ_FORCE_VARIANT must be'
        ' fast|no_basediff|no_smem_lut (got \'%s\')", e);\n'
        "    }\n"
        "  } else {\n"
        '    const char* smem_e = std::getenv("HIPVS_IVF_PQ_FORCE_SMEM_LUT");\n'
        '    const char* pre_e  = std::getenv("HIPVS_IVF_PQ_FORCE_PRECOMP");\n'
        "    if (smem_e != nullptr || pre_e != nullptr) {\n"
        "      const bool want_smem = (smem_e == nullptr) || (smem_e[0] != '0');\n"
        "      const bool want_pre  = (pre_e == nullptr) || (pre_e[0] != '0');\n"
        "      if (want_smem && want_pre) {\n"
        "        force_variant_idx = 0;\n"
        "      } else if (want_smem && !want_pre) {\n"
        "        force_variant_idx = 1;\n"
        "      } else {\n"
        "        // no_smem_lut always keeps PrecompBaseDiff=true in stock templates\n"
        "        force_variant_idx = 2;\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  uint32_t force_block_threads = 0;\n"
        '  if (const char* e = std::getenv("HIPVS_IVF_PQ_BLOCK_THREADS");'
        " e != nullptr && e[0] != '\\0') {\n"
        "    force_block_threads = static_cast<uint32_t>(std::atoi(e));\n"
        "    RAFT_EXPECTS(force_block_threads >= 32u &&"
        " (force_block_threads & (force_block_threads - 1u)) == 0u,\n"
        '                 "HIPVS_IVF_PQ_BLOCK_THREADS must be power-of-two >= 32");\n'
        "  }\n"
        "  {\n"
        "    static bool logged = false;\n"
        "    if (!logged && (force_variant_idx >= 0 || force_block_threads > 0 ||\n"
        '                    std::getenv("HIPVS_IVF_PQ_FORCE_NO_LOCAL_TOPK") != nullptr ||\n'
        '                    std::getenv("HIPVS_IVF_PQ_PREFERRED_CARVEOUT") != nullptr)) {\n'
        "      logged = true;\n"
        "      const char* vname =\n"
        '        force_variant_idx == 0 ? "fast" : force_variant_idx == 1 ? "no_basediff"\n'
        '                                                                : force_variant_idx == 2 ?'
        ' "no_smem_lut"\n'
        '                                                                                        : "(auto)";\n'
        "      std::fprintf(stderr,\n"
        '                   "[ann-harness] ivf_pq compute_similarity_select: variant=%s'
        ' block_threads=%u "\n'
        '                   "manage_local_topk=%d carveout=%g\\n",\n'
        "                   vname,\n"
        "                   force_block_threads,\n"
        "                   manage_local_topk ? 1 : 0,\n"
        "                   preferred_shmem_carveout);\n"
        "    }\n"
        "  }\n"
        "\n"
        "  // Shared memory for storing the lookup table\n"
    )
    if anchor not in t:
        raise SystemExit("select anchor missing")
    t = t.replace(anchor, insert, 1)

    old_for = (
        "  occupancy_t<OutT, LutT, IvfSampleFilterT> selected_perf{};\n"
        "  selected<OutT, LutT, IvfSampleFilterT> selected_config;\n"
        "  for (auto [kernel, smem_size_f, lut_is_in_shmem] : candidates) {\n"
        "    if (smem_size_f(dev_props.warpSize) > dev_props.sharedMemPerBlockOptin) {\n"
    )
    new_for = (
        "  occupancy_t<OutT, LutT, IvfSampleFilterT> selected_perf{};\n"
        "  selected<OutT, LutT, IvfSampleFilterT> selected_config;\n"
        "  int cand_idx = 0;\n"
        "  for (auto [kernel, smem_size_f, lut_is_in_shmem] : candidates) {\n"
        "    if (force_variant_idx >= 0 && cand_idx++ != force_variant_idx) { continue; }\n"
        "    if (force_variant_idx < 0) { ++cand_idx; }\n"
        "    if (smem_size_f(dev_props.warpSize) > dev_props.sharedMemPerBlockOptin) {\n"
    )
    if old_for not in t:
        raise SystemExit("for-loop anchor missing")
    t = t.replace(old_for, new_for, 1)

    old_reduce = (
        "    occupancy_t<OutT, LutT, IvfSampleFilterT> cur(smem_size, n_threads, kernel, dev_props);\n"
        "    if (cur.blocks_per_sm <= 0) {\n"
        "      // For some reason, we still cannot make this kernel run. Skip the candidate.\n"
        "      continue;\n"
        "    }\n"
        "\n"
        "    {\n"
        "      // Try to reduce the number of threads to increase occupancy and data locality\n"
        "      auto n_threads_tmp = n_threads_min;\n"
    )
    new_reduce = (
        "    occupancy_t<OutT, LutT, IvfSampleFilterT> cur(smem_size, n_threads, kernel, dev_props);\n"
        "    if (cur.blocks_per_sm <= 0) {\n"
        "      // For some reason, we still cannot make this kernel run. Skip the candidate.\n"
        "      continue;\n"
        "    }\n"
        "\n"
        "    if (force_block_threads > 0) {\n"
        "      uint32_t n_threads_forced =\n"
        "        raft::round_down_safe<uint32_t>(std::min(force_block_threads, n_threads),"
        " n_threads_gty);\n"
        "      if (n_threads_forced < n_threads_gty) { n_threads_forced = n_threads_gty; }\n"
        "      if (n_threads_forced > n_threads) { n_threads_forced = n_threads; }\n"
        "      auto smem_size_forced = smem_size_f(n_threads_forced);\n"
        "      cudaError_t st_forced =\n"
        "        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,"
        " smem_size_forced);\n"
        "      if (st_forced != cudaSuccess) {\n"
        "        RAFT_EXPECTS(st_forced == cudaGetLastError(),\n"
        '                     "Tried to reset the expected cuda error code,'
        ' but it didn\'t match the expectation");\n'
        "        continue;\n"
        "      }\n"
        "      occupancy_t<OutT, LutT, IvfSampleFilterT> forced(\n"
        "        smem_size_forced, n_threads_forced, kernel, dev_props);\n"
        "      if (forced.blocks_per_sm <= 0) { continue; }\n"
        "      n_threads = n_threads_forced;\n"
        "      smem_size = smem_size_forced;\n"
        "      cur       = forced;\n"
        "    } else {\n"
        "      // Try to reduce the number of threads to increase occupancy and data locality\n"
        "      auto n_threads_tmp = n_threads_min;\n"
    )
    if old_reduce not in t:
        raise SystemExit("reduce anchor missing")
    t = t.replace(old_reduce, new_reduce, 1)

    old_close = (
        "          n_threads_tmp /= 2;\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "\n"
        "    {\n"
        "      if (selected_perf.occupancy <= 0.0  // no candidate yet\n"
    )
    new_close = (
        "          n_threads_tmp /= 2;\n"
        "        }\n"
        "      }\n"
        "    }  // end auto thread-count tune (skipped when HIPVS_IVF_PQ_BLOCK_THREADS set)\n"
        "\n"
        "    {\n"
        "      if (selected_perf.occupancy <= 0.0  // no candidate yet\n"
    )
    if old_close not in t:
        raise SystemExit("close anchor missing")
    t = t.replace(old_close, new_close, 1)

    old_break = (
        "        RAFT_CUDA_TRY(\n"
        "          cudaFuncSetAttribute(kernel,"
        " cudaFuncAttributePreferredSharedMemoryCarveout, carveout));\n"
        "        if (cur.occupancy >= kTargetOccupancy) { break; }\n"
        "      } else if (selected_perf.occupancy > 0.0) {\n"
    )
    new_break = (
        "        RAFT_CUDA_TRY(\n"
        "          cudaFuncSetAttribute(kernel,"
        " cudaFuncAttributePreferredSharedMemoryCarveout, carveout));\n"
        "        if (force_variant_idx >= 0) { break; }\n"
        "        if (cur.occupancy >= kTargetOccupancy) { break; }\n"
        "      } else if (selected_perf.occupancy > 0.0) {\n"
    )
    if old_break not in t:
        raise SystemExit("break anchor missing")
    t = t.replace(old_break, new_break, 1)

    DST.write_text(t, encoding="utf-8")

    # Unified diff with hipVS-relative paths
    import difflib

    a = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
    b = t.splitlines(keepends=True)
    diff = difflib.unified_diff(
        a,
        b,
        fromfile="a/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh",
        tofile="b/cpp/src/neighbors/ivf_pq/ivf_pq_compute_similarity_impl.cuh",
        n=3,
    )
    PATCH.write_text("".join(diff), encoding="utf-8", newline="\n")
    print(f"wrote {DST.name} and {PATCH.name}")
    print(f"force_variant_idx count={t.count('force_variant_idx')}")
    print(f"force_block_threads count={t.count('force_block_threads')}")


if __name__ == "__main__":
    main()
