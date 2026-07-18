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
# If up fails with "container name already in use", remove orphans from another project:
docker rm -f milvus-etcd milvus-minio milvus-standalone 2>/dev/null || true
docker-compose up -d
sleep 20
docker-compose ps
# Expect etcd + minio + milvus-standalone all Up; standalone preferably (healthy)

# 3) If still unhealthy / rootcoord errors, wipe volumes and recreate
docker-compose down -v
docker rm -f milvus-etcd milvus-minio milvus-standalone 2>/dev/null || true
docker-compose up -d
sleep 30
docker logs milvus-standalone 2>&1 | tail -40
curl -sf http://127.0.0.1:9091/healthz && echo OK

# 4) Only then re-run MODE=cpu from nprobe=1
```

### 2) Switch to HIP Milvus

```bash
# Stop Docker milvus-standalone (free :19530); keep etcd/minio
docker stop milvus-standalone

export WORKDIR=~/rocmds_check_gfx1100
export ROCM_PATH=/opt/rocm
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export MILVUS_HIP_INSTALL_PREFIX=$WORKDIR/install
export LD_LIBRARY_PATH=$MILVUS_HIP_INSTALL_PREFIX/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}

# REQUIRED: rocksmq must NOT use /var/lib/milvus (permission denied as non-root)
mkdir -p "$WORKDIR/milvus_runtime"/{rdb_data,rdb_data_kv,data,logs}
cd "$WORKDIR/milvus"
# one-time (or if configs were reset): redirect paths under home
if grep -q '/var/lib/milvus' configs/milvus.yaml 2>/dev/null; then
  cp -a configs/milvus.yaml "configs/milvus.yaml.bak-$(date +%Y%m%d%H%M%S)"
  sed -i "s|/var/lib/milvus|$WORKDIR/milvus_runtime|g" configs/milvus.yaml
fi
grep -nE 'rocksmq|/var/lib|localStorage|path:' configs/milvus.yaml | head -30

pkill -f 'bin/milvus' || true
cd "$WORKDIR/milvus"
nohup env ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  ./bin/milvus run standalone >$WORKDIR/logs/milvus_gpu_standalone.log 2>&1 &

# wait — do not start VDBBench until OK
for i in $(seq 1 90); do
  curl -sf http://127.0.0.1:9091/healthz >/dev/null && echo "OK after ${i}s" && break
  sleep 2
done
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

## Search stage (important)

Default is now **concurrent** QPS (`SEARCH_STAGE=concurrent`):

```bash
MODE=cpu bash scripts/run_vdbbench_milvus_ivf_sweep.sh          # multi-client QPS
MODE=gpu bash scripts/run_vdbbench_milvus_ivf_sweep.sh
# optional: NUM_CONCURRENCY=1,10,20,40,80 CONCURRENCY_DURATION=30
# optional: SEARCH_STAGE=both   # serial recall + concurrent QPS
# optional: SEARCH_STAGE=serial # latency/recall only (old behaviour)
```

Earlier runs used serial by mistake for a “fair” latency recipe; that under-uses GPU.
Re-run CPU then GPU with the default concurrent stage for the management VDBBench table.

## Comparison A — VDBBench serial (2026-07-18)

Logs: `vdb_cpu_ivf_nprobe_20260718_194011.log`, `vdb_gpu_ivf_nprobe_20260718_195954.log`.

| nprobe | CPU QPS | CPU R@10 | GPU QPS | GPU R@10 | Speed-up |
|--------|---------|----------|---------|----------|----------|
| 1 | 1575 | 0.370 | 418 | 0.384 | 0.27× |
| 4 | 1211 | 0.702 | 412 | 0.707 | 0.34× |
| 8 | 924 | 0.839 | 345 | 0.841 | 0.37× |
| 16 | 626 | 0.931 | 676 | 0.931 | 1.08× |
| 32 | 371 | 0.979 | 773 | 0.979 | 2.08× |

Use for: recall correctness. Not the GPU speed headline.

## Comparison B — Batched 10k/search (throughput; lead with this)

Same client shape as Layer-4: `run_milvus_hdf5.py` one `search()` with all 10k queries.

| nprobe | CPU QPS† | GPU QPS (L4) | R@10 (GPU) | Speed-up |
|--------|----------|--------------|------------|----------|
| 1 | 22013 | 16982 | 0.382 | 0.77× |
| 4 | 14509 | 21882 | 0.709 | 1.51× |
| 8 | 9516 | 21838 | 0.844 | 2.30× |
| 16 | 5402 | 26181 | 0.934 | 4.85× |
| 32 | 2749 | 19578 | 0.980 | 7.12× |

†CPU from same-host batched diagnostic (runbook Table 7). GPU = Layer-4 harness.
For strict v2.5.4 parity, start Docker CPU and run:

```bash
# HIP stopped; Docker milvus-standalone healthy
cd ~/ann-harness-amd && git pull
bash scripts/run_milvus_batch_cpu_compare.sh
```

Then refresh Table B CPU column from the new JSON.
