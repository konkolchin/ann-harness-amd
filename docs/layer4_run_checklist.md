# Layer 4 run checklist (HIP Milvus GPU_IVF_FLAT, SIFT-1M)

Lab host: `amd-rx7900xtx`. Compare to runbook Tables 5 (FAISS) and 6–7 (CPU Milvus): `nlist=1024`, `k=10`, `nprobe=1,4,8,16,32`.

**Status: PASSED (2026-07-17)** — collection `sift_gpu_l4_20260717_200716`, JSON `$WORKDIR/logs/layer4_gpu_ivf_20260717_200716.json`. See `porting_milvus_gpu_to_amd.tex` Table layer4-measured.

| nprobe | QPS | p99 ms | recall@10 |
|--------|-----|--------|-----------|
| 1 | 16981.8 | 2.59 | 0.3824 |
| 4 | 21881.7 | 2.49 | 0.7091 |
| 8 | 21838.3 | 2.76 | 0.8436 |
| 16 | 26181.3 | 2.76 | 0.9337 |
| 32 | 19578.4 | 2.81 | 0.9803 |

Sealed path: IndexNode `GPU_CUVS_IVF_FLAT_*` (multi-slice, ~1M rows); QueryNode `CGO_LOAD GPU_CUVS_IVF_FLAT`; no `InvalidDeviceFunction`.

## Preflight

- [x] `data/sift-128-euclidean.hdf5` present under `~/ann-harness-amd/data/`
- [x] HIP Milvus built (`~/rocmds_check_gfx1100/milvus/bin/milvus`)
- [x] etcd + minio up (compose under milvus `deployments/docker/dev`)
- [x] Writable rocksmq/runtime paths (not `/var/lib/milvus`)
- [x] Discrete GPU pinned: `ROCR_VISIBLE_DEVICES=0` (hide iGPU `gfx1036`)
- [x] Enough disk for 1M IVF index + minio objects
- [x] Run in `tmux` / `screen` (insert+index can take tens of minutes+)

## Run

```bash
cd ~/ann-harness-amd && git pull --ff-only origin master

export WORKDIR=~/rocmds_check_gfx1100
export ROCM_PATH=/opt/rocm
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export MILVUS_HIP_INSTALL_PREFIX=$WORKDIR/install

# HIP milvus already on :19530 (default SKIP_START=1)
tmux new -s milvus-l4
bash scripts/run_milvus_layer4.sh

# Or start milvus from the script:
# SKIP_START=0 bash scripts/run_milvus_layer4.sh
```

Results JSON: `$WORKDIR/logs/layer4_gpu_ivf_*.json`  
Milvus log: `$WORKDIR/logs/milvus_gpu_standalone.log` (if started via harness)

## Pass criteria

- [x] `recall@10` **rises** with `nprobe` (FAISS-like ladder ~0.37 → ~0.98 at nprobe 1→32)
- [x] IndexNode log: `GPU_CUVS_IVF_FLAT_*` saved
- [x] QueryNode sealed load: `CGO_LOAD` `GPU_CUVS_IVF_FLAT` without `InvalidDeviceFunction`
- [x] Search is not only growing-path `IVF_FLAT_CC` (use `--flush`; layer4 script always flushes)
- [x] JSON results saved under `$WORKDIR/logs/`

## After the run

1. Fill Layer-4 measured table in `docs/porting_milvus_gpu_to_amd.tex` — done
2. Decide DXC merge of `amd-hip-gfx1100-layer3` on `llmkb-internal/milvus` — ready
