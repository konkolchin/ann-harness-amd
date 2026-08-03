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

No CAGRA Knowhere patch yet — waiting on lab `reproduce_cagra_gfx1100.sh` +
`classify_cagra_triage.py`.
