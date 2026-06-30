# ANN Harness (AMD-focused)

Reproducible baseline scripts for ANN testing with ready-to-use vectors (`sift-128-euclidean.hdf5`):

- `scripts/run_faiss_hdf5.py` — exact + ANN baseline in FAISS
- `scripts/run_milvus_hdf5.py` — Milvus baseline (Lite by default, supports `--uri`)
- `scripts/investigate_milvus_nprobe.py` — diagnose whether `nprobe` is effective in Milvus (`--uri` supported)
- `scripts/check_rocmds_gfx1030.sh` — environment/build check for hipRAFT + hipVS

## Documentation

- **Benchmarking runbook** (LaTeX): [`docs/ann_framework_runbook.tex`](docs/ann_framework_runbook.tex) — FAISS vs Milvus experiments, VectorDBBench, measured tables
- AMD GPU porting plan: [`docs/porting_milvus_gpu_to_amd.tex`](docs/porting_milvus_gpu_to_amd.tex)
- Runbook PDF (if compiled locally): [`docs/ANN_Testing_Framework (1).pdf`](docs/ANN_Testing_Framework%20(1).pdf)
- Quickstart (copy-paste): [`docs/QUICKSTART.md`](docs/QUICKSTART.md)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
wget --no-proxy https://ann-benchmarks.com/sift-128-euclidean.hdf5 -O data/sift-128-euclidean.hdf5
python scripts/run_faiss_hdf5.py
python scripts/run_milvus_hdf5.py
python scripts/investigate_milvus_nprobe.py
```

## Notes

- Milvus script uses local Milvus Lite DB: `./milvus_sift.db`.
- Insert is chunked to avoid gRPC max-message limit.
- `nprobe` sweeps are included for both FAISS and Milvus.
- If Milvus Lite shows no `nprobe` effect while FAISS does, verify on standalone Milvus via `--uri`.
