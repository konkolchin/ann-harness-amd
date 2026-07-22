# RTX 4080 — CUDA Milvus benchmarks (fair vs AMD HIP)

Goal: same client protocol as AMD Layer‑4 so we can compare HIP vs CUDA.

| Field | Value (freeze this) |
|-------|---------------------|
| Milvus | **v2.5.4 GPU** (`milvusdb/milvus:v2.5.4-gpu`) |
| Host | Ubuntu 22 + RTX 4080 |
| Dataset | SIFT-1M `sift-128-euclidean.hdf5` |
| Client | `scripts/run_milvus_hdf5.py` (batched 10k queries) |
| Seal | always `--flush` |
| `k` | 10 |
| `nlist` | 1024 |
| `nprobe` | 1,4,8,16,32 |

**Primary runs**

1. `GPU_IVF_FLAT`
2. `GPU_IVF_PQ` with **`m=32`**, `nbits=8` (best AMD PQ recall band)
3. Optional: `GPU_IVF_PQ` `m=16` (second point)

Do **not** mix batched QPS with VectorDBBench serial in one speed-up column.

---

## 0) Host preflight

```bash
nvidia-smi
docker --version
# NVIDIA Container Toolkit (required for --gpus)
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

If the last command fails, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Python client (host or venv):

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv wget
python3 -m venv ~/milvus-bench-venv
source ~/milvus-bench-venv/bin/activate
pip install -U pip
pip install 'pymilvus>=2.5,<2.6' h5py numpy
```

---

## 1) Clone harness + dataset

```bash
cd ~
git clone https://github.com/konkolchin/ann-harness-amd.git
cd ann-harness-amd
git pull --ff-only origin master

mkdir -p data logs
wget -c https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
  -O data/sift-128-euclidean.hdf5
```

---

## 2) Start CUDA Milvus GPU (Docker Compose)

```bash
cd ~/ann-harness-amd
export WORKDIR=~/milvus_cuda_4080
export MILVUS_IMAGE=milvusdb/milvus:v2.5.4-gpu
bash scripts/start_milvus_cuda_gpu_docker.sh
```

Health check:

```bash
curl -sf http://127.0.0.1:9091/healthz && echo OK
```

Stop later:

```bash
bash scripts/stop_milvus_cuda_gpu_docker.sh
```

---

## 3) Smoke (optional, small slice)

```bash
source ~/milvus-bench-venv/bin/activate
cd ~/ann-harness-amd
export WORKDIR=~/milvus_cuda_4080

# FLAT smoke
python3 scripts/run_milvus_hdf5.py \
  --uri http://127.0.0.1:19530 \
  --index-type GPU_IVF_FLAT --flush \
  --nlist 128 --nprobes 8,16 \
  --max-train-rows 50000 --max-query-rows 500 \
  --collection sift_cuda_flat_smoke \
  --results-json $WORKDIR/logs/smoke_flat.json

# PQ smoke (m=32)
python3 scripts/run_milvus_hdf5.py \
  --uri http://127.0.0.1:19530 \
  --index-type GPU_IVF_PQ --flush \
  --nlist 128 --m 32 --nbits 8 --nprobes 8,16 \
  --max-train-rows 50000 --max-query-rows 500 \
  --collection sift_cuda_pq32_smoke \
  --results-json $WORKDIR/logs/smoke_pq32.json
```

Smoke recall vs full‑1M GT is meaningless (same as AMD). Pass = builds/loads/searches.

Log check:

```bash
docker logs milvus-standalone 2>&1 | grep -aE 'GPU_CUVS_IVF_FLAT|GPU_CUVS_IVF_PQ|CUDA|error' | tail -40
```

---

## 4) Full SIFT Layer‑4 (real numbers)

Use `tmux`.

```bash
source ~/milvus-bench-venv/bin/activate
cd ~/ann-harness-amd
export WORKDIR=~/milvus_cuda_4080
mkdir -p $WORKDIR/logs

# A) GPU_IVF_FLAT
bash scripts/run_milvus_layer4_cuda.sh

# B) GPU_IVF_PQ m=32 (primary HIP↔CUDA PQ compare)
M=32 bash scripts/run_milvus_layer4_pq.sh
# (same script as AMD; only Milvus backend differs)
```

Expected JSON:

- `$WORKDIR/logs/layer4_cuda_gpu_ivf_*.json` (FLAT)
- `$WORKDIR/logs/layer4_gpu_ivf_pq_*.json` (PQ; rename/copy with `cuda` in the name if useful)

---

## 5) Compare to AMD (already measured)

**FLAT (AMD HIP batched)** — recall@10 / QPS at nprobe 16–32 ≈ 0.93–0.98 / ~20–26k.

**PQ m=32 (AMD HIP)**

| nprobe | QPS | recall@10 |
|--------|-----|-----------|
| 1 | 24273 | 0.36 |
| 4 | 20967 | 0.60 |
| 8 | 19500 | 0.67 |
| 16 | 22954 | 0.71 |
| 32 | 14784 | 0.73 |

On CUDA: match **recall ladder** first; then compare QPS at the same nprobe.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `could not select device driver` / no GPU in container | Install NVIDIA Container Toolkit; restart docker |
| Port 19530 busy | `docker ps`; stop old milvus/CPU stack |
| pymilvus version errors | Use `pymilvus` 2.5.x with Milvus 2.5.4 |
| Index builds as CPU | Confirm image tag ends with `-gpu`; check logs for `GPU_CUVS_*` |
