# Escalation: hipVS CAGRA filtered search broken on gfx1100 (RDNA3 consumer)

**File against:** [ROCm-DS/hipVS](https://github.com/ROCm-DS/hipVS/issues)  
**Also CC / mirror if needed:** ROCm-DS contacts, Milvus HIP porting thread  
**Date:** 2026-08-05  
**Severity:** High for filtered ANN / Milvus bitset path; **unfiltered CAGRA works**

---

## Title (suggested)

```
[gfx1100] CAGRA search with filters.from_bitset: recall@1 = 0 (unfiltered OK)
```

---

## Summary

On **AMD Radeon RX 7900 XTX (`gfx1100`)**, hipVS Python `cuvs.neighbors.cagra.search(..., filter=filters.from_bitset(...))` returns **wrong neighbors** (~40% filter) or **all empty / `0xFFFFFFFF`** (dense simple bitset), while the **same index unfiltered** achieves **recall@1 = 1.0**.

This is **not** a Knowhere/Milvus wiring bug: the failure reproduces with a **standalone hipVS Python** script (no Knowhere).

Knowhere Catch2 `GPU_CUVS_CAGRA` bitset sections show the same symptoms after forcing `build_algo=NN_DESCENT`.

---

## Environment

Collect on the lab (paste into the issue):

```bash
echo "host=$(hostname)"
rocm-smi --showproductname 2>/dev/null | head -20
rocminfo 2>/dev/null | grep -E 'Name:|Marketing|gfx' | head -20
echo "ROCm:"; cat /opt/rocm/.info/version 2>/dev/null || ls /opt/rocm*
source ~/hipvs-bench-venv/bin/activate
python3 - <<'PY'
import cuvs, cupy
print("cuvs", getattr(cuvs, "__version__", "?"))
print("cupy", cupy.__version__)
from cuvs.neighbors import cagra, filters
print("cagra", cagra)
print("filters", [a for a in dir(filters) if not a.startswith("_")])
PY
# hipVS / install provenance if known:
ls -la ~/rocmds_check_gfx1100/install 2>/dev/null | head
```

**Lab baseline (fill gaps when filing):**

| Item | Value |
|------|--------|
| Host | `amd-rx7900xtx` |
| GPU | Radeon RX 7900 XTX / `gfx1100` |
| ROCm | 7.0.2 (confirm with command above) |
| API | hipVS Python (`import cuvs` — cuVS-compatible) |
| Build algo | `nn_descent` (required; `ivf_pq` graph build throws on this GPU) |

---

## Minimal repro

Public harness: https://github.com/konkolchin/ann-harness-amd

```bash
git clone https://github.com/konkolchin/ann-harness-amd.git
cd ann-harness-amd
# activate venv that can: from cuvs.neighbors import cagra, filters
source ~/hipvs-bench-venv/bin/activate
export WORKDIR=~/rocmds_check_gfx1100   # or any writable log dir
bash scripts/run_hipvs_cagra_filter_repro.sh
```

Core script: `scripts/probe_cuvs_cagra_filter.py`

- Builds CAGRA with `build_algo=nn_descent`, `graph_degree=32`, `itopk_size=128`, `k=1`
- Cases: **unfiltered**, **~40% excluded** (`filters.from_bitset`), **64-row Simple Bitset** (Knowhere Catch2 byte pattern, polarity converted to cuVS: bit1=allowed)

---

## Results (2026-08-05, amd-rx7900xtx)

Artifact: `lib_hipvs_cagra_filter_20260805_232922.json`

| Case | recall@1 | Detail |
|------|---------:|--------|
| unfiltered | **1.00** | self-search ids `0,1,2,3…` |
| filter_40pct | **0.00** | real but wrong ids; `neg1=0`, no filter leaks |
| simple_bitset_64 | **0.00** | **all** neighbors `-1` / `0xFFFFFFFF` (`neg1=64/64`) |

Sample (filter_40pct): `pred=[2162]` vs `gt=[232]` for q0.

---

## Expected

Filtered search recall should be high (Catch2 thresholds: >0.7 at ~40% filter, ≥0.8 on simple bitset). NVIDIA cuVS peer with the same script is the contrast baseline (`scripts/run_cuvs_cagra_filter_repro.sh`).

---

## Related (separate tickets OK)

1. **CAGRA `build_algo=ivf_pq`** intermediate graph on gfx1100 throws RAFT `graph_core` invalid/duplicated neighbors — workaround: `nn_descent`.
2. Knowhere/Milvus Catch2 bitset fails are **symptoms** of (1)+(filter); unfiltered L2/Cosine/Hamming/serialize self-search are OK with `NN_DESCENT`.

---

## Ask

1. Confirm whether CAGRA + `bitset_filter` / Python `filters.from_bitset` is validated on **gfx1100** (consumer RDNA3), or only Instinct.
2. Fix or document unsupported filtered CAGRA on gfx1100.
3. If wavefront/bitset layout related, guidance on `USE_WARPSIZE_32` / known gfx1100 caveats.

---

## GitHub issue body (copy-paste)

```markdown
### Title
[gfx1100] CAGRA search with filters.from_bitset: recall@1 = 0 (unfiltered OK)

### Describe the bug
On RX 7900 XTX (gfx1100), hipVS Python CAGRA **unfiltered** search has recall@1 = 1.0, but the same index with `cuvs.neighbors.filters.from_bitset` yields recall@1 = 0.0:
- ~40% filter: wrong neighbor ids (not empty)
- dense 64-row bitset (Catch2 simple pattern): all neighbors `0xFFFFFFFF` / -1

Reproduces **without Knowhere** via https://github.com/konkolchin/ann-harness-amd `scripts/run_hipvs_cagra_filter_repro.sh`.

### Steps to reproduce
```bash
source ~/hipvs-bench-venv/bin/activate
git clone https://github.com/konkolchin/ann-harness-amd.git && cd ann-harness-amd
export WORKDIR=/tmp/hipvs_filter_repro
bash scripts/run_hipvs_cagra_filter_repro.sh
```

### Expected behavior
Filtered CAGRA recall comparable to unfiltered / cuVS CUDA peer.

### Actual behavior
| case | recall@1 |
|------|----------|
| unfiltered | 1.00 |
| filter_40pct | 0.00 |
| simple_bitset_64 | 0.00 (all -1) |

### Environment
- GPU: RX 7900 XTX / gfx1100
- ROCm: _(paste `cat /opt/rocm/.info/version`)_
- hipVS / cuvs Python: _(paste version)_
- build_algo: nn_descent (ivf_pq graph build throws on this GPU)

### Additional context
Knowhere Catch2 `GPU_CUVS_CAGRA` bitset sections match these symptoms after `build_algo=NN_DESCENT`. Unfiltered serialize/self-search OK.
```
