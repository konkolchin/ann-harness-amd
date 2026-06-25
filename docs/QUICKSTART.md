# Quickstart

This is the shortest reproducible flow for a new user.

## 0) System prerequisites (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
```

## 1) Clone repository

```bash
git clone https://github.com/konkolchin/ann-harness-amd.git
cd ann-harness-amd
```

## 2) Create and activate Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Download dataset

```bash
mkdir -p data
wget --no-proxy https://ann-benchmarks.com/sift-128-euclidean.hdf5 -O data/sift-128-euclidean.hdf5
```

## 4) Optional dataset sanity check

```bash
python - <<'PY'
import h5py
f = h5py.File("data/sift-128-euclidean.hdf5", "r")
print(list(f.keys()))
for k in f.keys():
    print(k, f[k].shape, f[k].dtype)
PY
```

Expected keys:

- `train`
- `test`
- `neighbors`
- `distances`

## 5) Run FAISS baseline

```bash
python scripts/run_faiss_hdf5.py
```

## 6) Run Milvus Lite baseline

```bash
python scripts/run_milvus_hdf5.py
```

Notes:

- Uses local Milvus Lite file DB: `milvus_sift.db`
- First insert/index can take time
- For standalone Milvus server, pass URI explicitly:

```bash
python scripts/run_milvus_hdf5.py --uri "http://localhost:19530"
```

## 7) Diagnose Milvus nprobe behavior

```bash
python scripts/investigate_milvus_nprobe.py
```

Or against standalone Milvus:

```bash
python scripts/investigate_milvus_nprobe.py --uri "http://localhost:19530"
```

This script prints:

- FAISS reference curve
- Milvus curves
- digest variability summary (whether `nprobe` changes returned neighbors)

## 7.1) Optional: start standalone Milvus with Docker

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.4.8 \
  milvus run standalone
```

Then use `--uri "http://localhost:19530"` in Milvus scripts.

## 8) Common issues

### Download blocked by proxy/firewall

Try `--no-proxy` (already used above). If still blocked, download on another machine and copy file into `data/`.

### Milvus insert error: message too large

Reduce `INSERT_BATCH` in `scripts/run_milvus_hdf5.py` (e.g. `20000`).

### Milvus Lite: recall@10 unchanged across nprobe

If FAISS responds to `nprobe` but Milvus Lite shows identical recall and identical
result digests across the full sweep, treat this as a Lite/API behavior issue and
re-run diagnostics against standalone Milvus (`--uri http://localhost:19530`).

### Wrong directory

Run commands from repository root:

```bash
ann-harness-amd/
```
