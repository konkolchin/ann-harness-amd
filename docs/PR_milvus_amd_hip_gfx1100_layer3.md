# Layer 3: Milvus HIP GPU_IVF_FLAT on gfx1100

## Summary

- Wire Milvus GPU builds to DXC Knowhere (`2.5`) / optional `MILVUS_KNOWHERE_SOURCE_DIR`, with `WITH_HIP` + `WITH_CUVS` against prebuilt hipVS/hipRAFT.
- Fix HIP install-prefix / xxHash discovery for nested Knowhere; add Knowhere API and clang 19 Tantivy compat for Milvus `v2.5.4`.
- Validated on **RX 7900 XTX (gfx1100)**: sealed `GPU_IVF_FLAT` → IndexNode `GPU_CUVS_IVF_FLAT_*`, QueryNode sealed load without `hipErrorInvalidDeviceFunction` (use `ROCR_VISIBLE_DEVICES=0` to hide iGPU).

## Base

| Item | Value |
|------|--------|
| Branch | `amd-hip-gfx1100-layer3` |
| Against | DXC `llmkb-internal/milvus` default (`master` / `main` — confirm on remote) |
| Tag base | Milvus `v2.5.4` |
| DXC repo | https://github.dxc.com/llmkb-internal/milvus |

## Test plan

- [x] Build Milvus GPU (`v2.5.4`) against local/DXC HIP Knowhere + hipVS under `~/rocmds_check_gfx1100/install`
- [x] Standalone start with writable rocksmq/runtime paths (not `/var/lib/milvus`)
- [x] Client smoke: insert → **flush** → `GPU_IVF_FLAT` → load (`indexed_rows` complete, `LoadState: Loaded`) → search
- [x] Logs: IndexNode saves `GPU_CUVS_IVF_FLAT_*`; QueryNode `CGO_LOAD` `GPU_CUVS_IVF_FLAT`; no `InvalidDeviceFunction`
- [ ] Layer 4: full SIFT-1M recall/QPS vs CPU baseline

## Depends on

- [`llmkb-internal/knowhere`](https://github.dxc.com/llmkb-internal/knowhere) branch `2.5` (Layer-2 HIP, PR #2 merged)
- Runtime: pin discrete GPU (`ROCR_VISIBLE_DEVICES=0`); flush sealed segments before treating search as GPU (growing path still uses CPU `IVF_FLAT_CC`)

## Notes

- Companion harness: [`konkolchin/ann-harness-amd`](https://github.com/konkolchin/ann-harness-amd) (`build_milvus_layer3.sh`, milvus patches `0001`–`0004`, smoke with `--flush`)
- Knowhere follow-up recommended if not landed: `CMAKE_CURRENT_SOURCE_DIR` for nested `cuvs_knowhere_index_hip.cu` under Milvus
