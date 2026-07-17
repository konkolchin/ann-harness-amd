# Layer 4 run checklist (HIP Milvus GPU_IVF_FLAT, SIFT-1M)

Lab host: `amd-rx7900xtx`. Compare to runbook Tables 5 (FAISS) and 6–7 (CPU Milvus): `nlist=1024`, `k=10`, `nprobe=1,4,8,16,32`.

## Preflight

- [ ] `data/sift-128-euclidean.hdf5` present under `~/ann-harness-amd/data/`
- [ ] HIP Milvus built (`~/rocmds_check_gfx1100/milvus/bin/milvus`)
- [ ] etcd + minio up (compose under milvus `deployments/docker/dev`)
- [ ] Writable rocksmq/runtime paths (not `/var/lib/milvus`)
- [ ] Discrete GPU pinned: `ROCR_VISIBLE_DEVICES=0` (hide iGPU `gfx1036`)
- [ ] Enough disk for 1M IVF index + minio objects
- [ ] Run in `tmux` / `screen` (insert+index can take tens of minutes+)

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

- [ ] `recall@10` **rises** with `nprobe` (FAISS-like ladder ~0.37 → ~0.98 at nprobe 1→32)
- [ ] IndexNode log: `GPU_CUVS_IVF_FLAT_*` saved
- [ ] QueryNode sealed load: `CGO_LOAD` `GPU_CUVS_IVF_FLAT` without `InvalidDeviceFunction`
- [ ] Search is not only growing-path `IVF_FLAT_CC` (use `--flush`; layer4 script always flushes)
- [ ] JSON results saved under `$WORKDIR/logs/`

## After the run

1. Fill Layer-4 measured table in `docs/porting_milvus_gpu_to_amd.tex`
2. Decide DXC merge of `amd-hip-gfx1100-layer3` on `llmkb-internal/milvus`
