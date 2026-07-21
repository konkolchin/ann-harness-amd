# IVF_PQ on AMD (today) → CUDA 4080 (tomorrow)

Fair compare needs **identical** recipe on both GPUs:

| Field | Value |
|-------|--------|
| Dataset | SIFT-1M (`sift-128-euclidean.hdf5`) |
| Index | `GPU_IVF_PQ` |
| `nlist` | `1024` (full) / `128` (smoke) |
| `m` | `16` (must divide 128) |
| `nbits` | `8` |
| `k` | `10` |
| `nprobe` | `1,4,8,16,32` |
| Client | batched harness (`run_milvus_hdf5.py`, one search with all queries) |
| Seal | always `--flush` |

CPU peer later (optional): `IVF_PQ` with the same `nlist/m/nbits/nprobe` — not FLAT.

## Today — AMD RX 7900 XTX (`amd-rx7900xtx`)

```bash
cd ~/ann-harness-amd && git pull
export WORKDIR=~/rocmds_check_gfx1100
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0

# HIP milvus already on :19530
bash scripts/run_milvus_gpu_ivf_pq_smoke.sh
```

Pass smoke if:

- client finishes without exception
- recall@10 at `nprobe` 8/16 is **below** FLAT but **> 0** and rises with nprobe
- milvus log shows sealed `GPU_CUVS_IVF_PQ` (not only growing-path remap)
- no `InvalidDeviceFunction`

Then full SIFT (long; use `tmux`):

```bash
bash scripts/run_milvus_layer4_pq.sh
# JSON: $WORKDIR/logs/layer4_gpu_ivf_pq_*.json
```

Known risk (from Knowhere UT notes): occasional IVF_PQ TopK threshold flakes on gfx1100 — if smoke recall is nonsense (0) or build crashes, capture log + stop before full 1M.

## Tomorrow — NVIDIA 4080

Same commands against CUDA Milvus GPU build / image, **same** `NLIST/M/NBITS/NPROBES`. Compare QPS and recall@10 side by side; do not mix batched vs serial in one speed-up column.

## What “good” looks like vs FLAT

- PQ recall@10 usually **lower** than FLAT at the same nprobe (compression tax).
- PQ QPS often **higher** and memory much **lower** — that is the product story.
- HIP vs CUDA: match recall ladders first; then compare QPS at the same nprobe / same recall band.
