# Dashboard / CI proposal — `amd-cagra`

Companion to `docs/cagra_consumer_followon.md` for the internal Milvus AMD GPU
parity board (and GitHub Actions when wired).

## Principles

1. Do not fail NVIDIA CUDA jobs when HIP is unset (`WITH_HIP` / `MILVUS_WITH_HIP` opt-in only).
2. Separate **lib** evidence from **product** evidence (same as IVF).
3. CAGRA Catch2 red must not erase IVF Layer-2 gate PASS — filter or split rows.

## Proposed rows

| Row id | Side | Artifact | Pass |
|--------|------|----------|------|
| `amd-cagra-catch2` | AMD | Knowhere `-k CAGRA` log | TopK + Serialize PASS |
| `amd-cagra-lib` | AMD | `lib_hipvs_cagra_*.json` | max recall@10 ≥ 0.7 at itopk≥128 |
| `nvidia-cagra-lib` | NVIDIA | `lib_cuvs_cagra_*.json` | peer baseline |
| `cagra-lib-parity` | both | compare script | recall Δ ≤ 0.05; QPS ratio recorded |
| `amd-cagra-smoke` | AMD | smoke client + milvus log | sealed `GPU_CUVS_CAGRA`; recall > 0 |
| `amd-cagra-l4` | AMD | `layer4_gpu_cagra_*.json` | recall rises with itopk |
| `nvidia-cagra-l4` | NVIDIA | same recipe | peer |
| `cagra-product-parity` | both | L4 pair | recall Δ gate; QPS informational |

## Job sketch (pseudo)

```yaml
# Only on HIP-capable runners / when MILVUS_WITH_HIP=1
cagra-phase-a:
  steps:
    - run: bash scripts/reproduce_cagra_gfx1100.sh
    - run: bash scripts/run_hipvs_cagra_bench.sh
    - upload: logs/lib_hipvs_cagra_*.json

cagra-phase-b:
  needs: cagra-phase-a  # or manual after Catch2 green
  steps:
    - run: bash scripts/run_milvus_gpu_cagra_smoke.sh
    - run: SKIP_START=1 bash scripts/run_milvus_layer4_cagra.sh
```

## Exact-SHA note

If PR `ci/exact-sha-gpu-build` pins Knowhere/hipVS SHAs, include CAGRA-capable
SHAs only after Phase A lands; short GHA failures (~6–11m) are often cancel/setup —
use the long GPU build run for diagnosis.
