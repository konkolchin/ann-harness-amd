# Fix: WITH_HIP must be explicit (CUDA CI / dashboard BLOCKED)

## Problem

Two places forced or auto-enabled HIP and broke clean NVIDIA builds:

1. **Knowhere** auto-set `WITH_HIP=ON` when `INSTALL_PREFIX` contained
   `lib/cmake/cuvs/cuvs-config.cmake` (true for CUDA cuVS too).
2. **Milvus Layer-3 patch** did `set(WITH_HIP ON CACHE BOOL "" FORCE)` whenever
   `MILVUS_GPU_VERSION=ON` — so any GPU CI tree pulled ROCm/hipVS.

Dashboard showed NVIDIA product benches as **BLOCKED** (build), not a bad QPS result.

## Fix

### Knowhere (`WITH_HIP` opt-in only)

PR: https://github.com/konkolchin/knowhere/pull/1  
Branch: `fix/with-hip-explicit-opt-in`

- Default `WITH_HIP=OFF`
- Enable only with `-DWITH_HIP=ON` or `KNOWHERE_WITH_HIP=1|ON|TRUE|YES`
- **No** auto-detect from `INSTALL_PREFIX`

Harness patch: [`0049-with-hip-explicit-opt-in-only.patch`](../patches/knowhere/0049-with-hip-explicit-opt-in-only.patch)

### Milvus thirdparty Knowhere CMake

[`0001-knowhere-hip-dxc-fetchcontent.patch`](../patches/milvus/0001-knowhere-hip-dxc-fetchcontent.patch):

- Still sets `WITH_CUVS=ON` for GPU builds
- Sets `WITH_HIP=ON` **only** if `-DMILVUS_WITH_HIP=ON` or env `MILVUS_WITH_HIP` / `KNOWHERE_WITH_HIP` is truthy
- HIP `CMAKE_PREFIX_PATH` / xxHash paths apply only when `WITH_HIP` is on

## Apply on DXC Knowhere (`llmkb-internal/knowhere` branch `2.5`)

```bash
cd /path/to/llmkb-internal/knowhere
git fetch https://github.com/konkolchin/knowhere.git fix/with-hip-explicit-opt-in
git cherry-pick f2083f4
# or apply: patches/knowhere/0049-with-hip-explicit-opt-in-only.patch
```

Re-apply / refresh Milvus Layer-3 patches from this harness so `0001` no longer FORCE-ONs HIP.

## AMD lab builds (must opt in)

```bash
export MILVUS_WITH_HIP=1
export KNOWHERE_WITH_HIP=1
export INSTALL_PREFIX=~/rocmds_check_gfx1100/install
# Layer 2:
cmake ... -DWITH_CUVS=ON -DWITH_HIP=ON
# Layer 3 (build_milvus_layer3.sh already exports these):
bash scripts/build_milvus_layer3.sh
```

## CUDA / NVIDIA CI

```bash
unset MILVUS_WITH_HIP KNOWHERE_WITH_HIP
# do not pass -DMILVUS_WITH_HIP=ON or -DWITH_HIP=ON
cmake ... -DWITH_CUVS=ON   # WITH_HIP stays OFF
```

Clear any stale CMake cache that previously forced `WITH_HIP:BOOL=ON`.
