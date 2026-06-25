# ANN Harness (AMD-focused)

Reproducible baseline scripts for ANN testing with ready-to-use vectors (`sift-128-euclidean.hdf5`):

- `scripts/run_faiss_hdf5.py` — exact + ANN baseline in FAISS
- `scripts/run_milvus_hdf5.py` — Milvus Lite baseline (QPS, p99, recall@10)
- `scripts/check_rocmds_gfx1030.sh` — environment/build check for hipRAFT + hipVS

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
wget --no-proxy https://ann-benchmarks.com/sift-128-euclidean.hdf5 -O data/sift-128-euclidean.hdf5
python scripts/run_faiss_hdf5.py
python scripts/run_milvus_hdf5.py
```

## Notes

- Milvus script uses local Milvus Lite DB: `./milvus_sift.db`.
- Insert is chunked to avoid gRPC max-message limit.
- `nprobe` sweeps are included for both FAISS and Milvus.
