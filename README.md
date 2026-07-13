# ANN Harness (AMD-focused)

Reproducible baseline scripts for ANN testing with ready-to-use vectors (`sift-128-euclidean.hdf5`):

- `scripts/run_faiss_hdf5.py` — exact + ANN baseline in FAISS
- `scripts/run_milvus_hdf5.py` — Milvus baseline (Lite by default, supports `--uri`)
- `scripts/investigate_milvus_nprobe.py` — diagnose whether `nprobe` is effective in Milvus (`--uri` supported)
- `scripts/check_rocmds_gfx1030.sh` — Layer 1 build check for hipRAFT + hipVS (6800 XT / gfx1030)
- `scripts/check_rocmds_gfx1100.sh` — Layer 1 build check for 7900 XTX / gfx1100 lab host
- `scripts/build_knowhere_layer2.sh` / `run_knowhere_gpu_tests.sh` — Layer 2 Knowhere HIP
- `scripts/build_milvus_layer3.sh` / `run_milvus_gpu_smoke.sh` — Layer 3 Milvus + HIP Knowhere
- `scripts/apply_hipvs_packer_debug_patch.sh` — Layer 1.5 debug patch only (legacy)
- `scripts/apply_hipvs_layer15_patches.sh` — apply debug + kIndexGroupSize fix patches to hipVS
- `scripts/check_ivf_packer_mismatch.sh` — check gtest log for `testPacker mask mismatch` lines

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
# For nprobe diagnostics, prefer standalone Milvus:
python scripts/investigate_milvus_nprobe.py --uri "http://127.0.0.1:19530"
```

## Notes

- Milvus Lite (`./milvus_sift.db`) is a smoke/bring-up path for insert/index/search plumbing.
- `nprobe` tuning should be validated on standalone Milvus (`--uri "http://127.0.0.1:19530"`).
- Standalone Milvus on `amd-rx7900xtx` (2026-07-07) showed expected behavior:
  recall@10 rose from `0.3901` (`nprobe=1`) to `0.9956` (`nprobe=64`) while QPS decreased.
- Standalone Milvus needs Docker on a host with **8 GB+ RAM**; use `--uri` from a
  smaller client machine (see `docs/QUICKSTART.md` §7.2).
- Insert is chunked to avoid gRPC max-message limit.
