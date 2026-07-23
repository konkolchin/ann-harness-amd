# Library bench results (hipVS vs cuVS)

JSON files from lab runs should be copied here (or keep under `$WORKDIR/logs/` and
point `compare_cuvs_lib_json.py` at those paths).

## Status (agent environment)

Lab hosts (`amd-rx7900xtx`, `kvkol-B650-AORUS-PRO-AX`) are **not reachable** from
the Cursor agent (no SSH). Runs must be executed on the boxes:

| Host | Command |
|------|---------|
| AMD RX 7900 XTX | `bash scripts/run_hipvs_ivf_bench.sh` then `INDEX_TYPE=IVF_PQ M=32 bash scripts/run_hipvs_ivf_bench.sh` |
| NVIDIA RTX 4080 | `WORKDIR=~/milvus_cuda_4080 bash scripts/run_cuvs_ivf_bench.sh` then PQ similarly |

See [hipvs_vs_cuvs_bench.md](../docs/hipvs_vs_cuvs_bench.md).

## Expected filenames

- `lib_hipvs_ivf_flat_YYYYMMDD_HHMMSS.json`
- `lib_hipvs_ivf_pq_m32_YYYYMMDD_HHMMSS.json`
- `lib_cuvs_ivf_flat_YYYYMMDD_HHMMSS.json`
- `lib_cuvs_ivf_pq_m32_YYYYMMDD_HHMMSS.json`

After both sides exist, fill `FILLED.md` with the compare script output.
