# Library bench results (hipVS vs cuVS)

JSON from lab runs live here.

## Filled compares

| File | What |
|------|------|
| [FILLED_pq_m32.md](FILLED_pq_m32.md) | IVF_PQ m=32 — hip ~0.5× cu (2026-07-23) |
| [FILLED_flat_heavy.md](FILLED_flat_heavy.md) | IVF_FLAT nprobe 32..256 — hip ~1.3× cu (2026-07-27) |
| [CAGRA_STATUS.md](CAGRA_STATUS.md) | CAGRA follow-on — Catch2 baseline + lab fill slots |

## How to re-run

| Host | Command |
|------|---------|
| AMD RX 7900 XTX | `INDEX_TYPE=IVF_PQ M=32 bash scripts/run_hipvs_ivf_bench.sh` |
| AMD (GPU-heavy FLAT) | `INDEX_TYPE=IVF_FLAT NPROBES=32,64,128,256 P99_SAMPLE=0 bash scripts/run_hipvs_ivf_bench.sh` |
| AMD CAGRA | `bash scripts/run_hipvs_cagra_bench.sh` |
| NVIDIA RTX 4080 | `WORKDIR=~/milvus_cuda_4080 INDEX_TYPE=… bash scripts/run_cuvs_ivf_bench.sh` |
| NVIDIA CAGRA | `WORKDIR=~/milvus_cuda_4080 bash scripts/run_cuvs_cagra_bench.sh` |

See [hipvs_vs_cuvs_bench.md](../docs/hipvs_vs_cuvs_bench.md) and
[cagra_consumer_followon.md](../docs/cagra_consumer_followon.md).
