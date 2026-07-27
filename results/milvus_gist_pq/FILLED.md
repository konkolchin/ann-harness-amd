# Milvus GIST GPU_IVF_PQ — AMD vs CUDA

**Status: lab pending** (no SSH to lab from agent). Fill after both sides run
`scripts/run_milvus_layer4_gist_pq.sh`.

| nprobe | AMD QPS (7900) | CUDA QPS (4080) | R@10 AMD | R@10 CUDA | AMD/CUDA |
|--------|----------------|-----------------|----------|-----------|----------|
| 8 | — | — | — | — | — |
| 16 | — | — | — | — | — |
| 32 | — | — | — | — | — |
| 64 | — | — | — | — | — |
| 128 | — | — | — | — | — |

```bash
python3 scripts/compare_milvus_layer4_json.py \
  --amd  results/milvus_gist_pq/<amd>.json \
  --cuda results/milvus_gist_pq/<cuda>.json \
  --out-md results/milvus_gist_pq/FILLED.md
```

Success target: matched recall (Δ ≤ 0.01) and **AMD/CUDA ≤ ~0.85×**.
