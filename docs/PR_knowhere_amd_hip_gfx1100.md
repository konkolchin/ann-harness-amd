# Add HIP/ROCm Layer-2 support for gfx1100 (AMD/DXC port)

## Summary

- Add HIP/ROCm Layer-2 support so Knowhere can build and run against prebuilt **hipVS / hipRAFT** (instead of NVIDIA cuVS/RAFT) on AMD GPUs.
- Host-safe HIP CMake path: strip HIP-only flags from host C++, link `libcuvs`/`libraft`/`rmm`, and fix logger/spdlog embedding for `libknowhere.so`.
- Validated on **RX 7900 XTX (gfx1100)**: Catch2 section `Test Gpu Index Search` (L2) — **108 assertions passed**.

## Base

| Item | Value |
|------|--------|
| Branch | `amd-hip-gfx1100-layer2` |
| Against | `2.5` |
| Commit | `3b21a04d` — *Add HIP/ROCm Layer-2 support for gfx1100 (AMD/DXC port).* |
| DXC repo | https://github.dxc.com/llmkb-internal/knowhere/ |
| Public mirror | https://github.com/konkolchin/knowhere/tree/amd-hip-gfx1100-layer2 |

## What changed

- New CMake: `libhipcuvs*.cmake`, `knowhere_hip_host_fixup.cmake`, `knowhere_hip_link.cmake`
- HIP shims: `cuda_compat.hpp`, GPU device-count helper, single HIP cuVS instantiation unit
- Logger/spdlog: host-only logger static lib + `SPDLOG_HEADER_ONLY` so `set_pattern` resolves at runtime
- Wire-up in `CMakeLists.txt`, GPU index sources, and UT CMake

## How to build / test (lab host)

```bash
# Prebuilt hipVS/hipRAFT under INSTALL_PREFIX, then:
export WORKDIR=~/rocmds_check_gfx1100
export INSTALL_PREFIX=$WORKDIR/install
export ROCM_PATH=/opt/rocm

# From harness (or equivalent Conan + cmake WITH_HIP=ON):
bash scripts/build_knowhere_layer2.sh
bash scripts/run_knowhere_gpu_tests.sh
# → Test All GPU Index --section 'Test Gpu Index Search'
```

## Test plan

- [x] Configure/build Knowhere with `WITH_HIP=ON` + prebuilt hipVS/hipRAFT on gfx1100
- [x] `knowhere_tests` section **Test Gpu Index Search** (L2) — 108 assertions green
- [ ] Full GPU suite (`Test All GPU Index`) — known gaps on gfx1100: CAGRA bitset recall=0; occasional IVF_PQ TopK threshold miss (~0.84 vs 0.85)
- [ ] Layer 3: Milvus linked against this Knowhere build

## Notes

- Layer-2 pass criterion for this PR is **L2 GPU search**, not the full Catch2 GPU suite.
- Companion harness/docs: [`konkolchin/ann-harness-amd`](https://github.com/konkolchin/ann-harness-amd) (patches/scripts used to develop this tree).
