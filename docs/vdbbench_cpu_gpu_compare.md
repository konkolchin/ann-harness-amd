# Fair Milvus CPU vs GPU HIP compare (VectorDBBench)

Apples-to-apples: **both** sides use VectorDBBench on `amd-rx7900xtx`, same SIFT-1M parquet, `nlist=1024`, `k=10`, `nprobe=1,4,8,16,32`.

| Side | CLI | Server on `:19530` | Index |
|------|-----|--------------------|--------|
| CPU | `milvusivfflat` | Docker Milvus CPU `v2.5.4` | `IVF_FLAT` |
| GPU | `milvusgpuivfflat` | HIP Standalone `v2.5.4` | `GPU_IVF_FLAT` |

Do **not** mix harness `run_milvus_layer4.sh` GPU numbers with VDBBench CPU numbers in management tables.

## Preflight

- [ ] `python3.11` venv: `~/vdbbench-venv` with `vectordb-bench` (`pip install -U vectordb-bench`)
- [ ] Parquet dataset at `~/vdbbench-sift1m/` (`train.parquet`, `test.parquet`, `neighbors.parquet`)
- [ ] ≥20–30 GB free on `/`
- [ ] Only **one** Milvus on `:19530` at a time
- [ ] GPU run: `ROCR_VISIBLE_DEVICES=0` (hide iGPU)

### Parquet prep (if missing)

Same as runbook § `vectordbbench-ivf-nprobe`:

```bash
source ~/vdbbench-venv/bin/activate
pip install h5py pandas pyarrow
mkdir -p ~/ann-harness-amd/data ~/vdbbench-sift1m
# ensure data/sift-128-euclidean.hdf5 exists, then convert (see runbook heredoc)
```

## Lab order (sequential)

### 1) CPU sweep (Docker Milvus)

```bash
cd ~/ann-harness-amd && git pull
source ~/vdbbench-venv/bin/activate

# Stop HIP milvus if it holds :19530; start Docker CPU Milvus v2.5.4
# This host uses classic docker-compose (hyphen), not `docker compose`:
#   pkill -f 'bin/milvus' || true
#   cd ~/ann-harness-amd/milvus-docker && docker-compose up -d
# Wait until healthy (standalone Up (healthy) + healthz):
docker-compose -f ~/ann-harness-amd/milvus-docker/docker-compose.yml ps
curl -sf http://127.0.0.1:9091/healthz && echo OK

tmux new -s vdb-cpu
MODE=cpu bash scripts/run_vdbbench_milvus_ivf_sweep.sh
```

Log: `logs/vdb_cpu_ivf_nprobe_*.log`

**Known failure A:** insert of 1M succeeds, then Milvus dies during **optimize/compact**
(`Connection refused` on `:19530`). VectorDBBench may still print a fake “Success” /
`qps=0` / label `x`. The sweep script now **exits non-zero** in that case.

**Known failure B:** logs show `find no available rootcoord` / `empty grpc client` —
etcd/minio/standalone are out of sync (partial crash). `:19530` and `:9091` stay down.

Recovery (CPU Docker — wipe is OK; VDBBench re-inserts):

```bash
# 1) Free ports: stop HIP standalone if running
pkill -f 'bin/milvus' || true
ss -lntp | grep -E '19530|9091' || true

# 2) Restart full stack (this lab uses docker-compose with hyphen)
cd ~/ann-harness-amd/milvus-docker
docker-compose ps
docker-compose down
docker-compose up -d
sleep 20
docker-compose ps
# Expect etcd + minio + milvus-standalone all Up; standalone preferably (healthy)

# 3) If still unhealthy / rootcoord errors, wipe volumes and recreate
docker-compose down -v
docker-compose up -d
sleep 30
docker logs milvus-standalone 2>&1 | tail -40
curl -sf http://127.0.0.1:9091/healthz && echo OK

# 4) Only then re-run MODE=cpu from nprobe=1
```

### 2) Switch to HIP Milvus

```bash
# Stop Docker milvus-standalone (free :19530)
docker stop milvus-standalone   # or your compose down for milvus only

export WORKDIR=~/rocmds_check_gfx1100
export ROCM_PATH=/opt/rocm
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export MILVUS_HIP_INSTALL_PREFIX=$WORKDIR/install
export LD_LIBRARY_PATH=$MILVUS_HIP_INSTALL_PREFIX/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}

# Start HIP milvus (same pattern as Layer-3/4); ensure etcd/minio up
cd $WORKDIR/milvus
nohup env ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  ./bin/milvus run standalone >$WORKDIR/logs/milvus_gpu_standalone.log 2>&1 &
```

### 3) GPU sweep

```bash
cd ~/ann-harness-amd
source ~/vdbbench-venv/bin/activate
tmux new -s vdb-gpu
MODE=gpu bash scripts/run_vdbbench_milvus_ivf_sweep.sh
```

Log: `logs/vdb_gpu_ivf_nprobe_*.log`

Optional GPU flags (defaults in script):

```bash
CACHE_ON_DEVICE=false REFINE_RATIO=1.0 MODE=gpu bash scripts/run_vdbbench_milvus_ivf_sweep.sh
```

## Parse results

Serial search often leaves summary `qps=0`. Use:

```bash
grep -E 'nprobe=|search entire test_data' logs/vdb_cpu_ivf_nprobe_*.log
grep -E 'nprobe=|search entire test_data' logs/vdb_gpu_ivf_nprobe_*.log
```

**QPS** = `queries / cost` from each `search entire test_data` line.  
**recall@10** = `avg_recall` (with `--k 10`).

## GPU sealed-path check

```bash
grep -aE 'GPU_CUVS_IVF_FLAT|InvalidDeviceFunction|IVF_FLAT_CC' \
  ~/rocmds_check_gfx1100/logs/milvus_gpu_standalone.log | tail -40
```

Pass: IndexNode/QueryNode activity for `GPU_CUVS_IVF_FLAT`; no `InvalidDeviceFunction`.

## Fill docs

Update Table `layer4-cpu-gpu` in `docs/porting_milvus_gpu_to_amd.tex` and the Results slide in `docs/porting_milvus_amd_slides.tex` with **paired** VDBBench CPU + GPU numbers and QPS speed-up = GPU/CPU.
