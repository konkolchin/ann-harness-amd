# Knowhere patches for CAGRA on gfx1100

Phase A triage may produce HIP-specific fixes for `GPU_CUVS_CAGRA`.

## Convention

- Next free number after `0051-…` (e.g. `0052-cagra-….patch`).
- Prefer minimal diffs; do not force `WITH_HIP` on CUDA CI.
- Document root cause in `docs/cagra_consumer_followon.md` status log.

## When patches belong here vs hipVS

| Symptom | Land where |
|---------|------------|
| `cuvs.neighbors.cagra` Python/C++ recall ~0 | hipVS / ROCm-DS (not Knowhere) |
| hipVS OK, Catch2 `GPU_CUVS_CAGRA` recall 0 | **this directory** |
| Serialize/Deserialize only | Knowhere index IO / HIP dtype |

## 0052 — Catch2 `cagra_gen` → `NN_DESCENT`

`patches/knowhere/0052-cagra-default-nn-descent-in-gpu-ut.patch`

Default IVF_PQ graph build → recall 0.0 on gfx1100; `NN_DESCENT` matches lib smoke
(recall 1.0). Hamming path already used `ITERATIVE` and passed.

```bash
bash scripts/apply_knowhere_cagra_nn_descent_ut.sh
RUN_CATCH2=1 bash scripts/apply_knowhere_cagra_nn_descent_ut.sh
```
