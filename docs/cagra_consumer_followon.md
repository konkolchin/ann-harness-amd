# CAGRA follow-on — consumer gfx1100 (A then B)

**Story:** AMD docs emphasize Instinct (`gfx90a` / `gfx942`). This follow-on
proves **consumer RDNA3** (lab: RX 7900 XTX / `gfx1100`) can run graph ANN
(`GPU_CAGRA`) in the same Milvus HIP stack that already ships IVF_FLAT / IVF_PQ.

**Phasing:** Phase **A** (library / Knowhere) before Phase **B** (Milvus product).

| Phase | Exit |
|-------|------|
| A | Catch2 CAGRA TopK + Serialize green (or escalated hipVS repro); lib JSON hipVS vs cuVS |
| B | Sealed `GPU_CAGRA` smoke + Layer-4 SIFT (GIST if VRAM) vs RTX 4080 |

IVF PoC remains valid; CAGRA was explicitly out of PoC scope (Catch2 recall **0.0**).

---

## Known baseline (2026-07-26, amd-rx7900xtx)

Full Knowhere Catch2 `Test All GPU Index`: **579/582** — three fails, two of them CAGRA:

| Section | Index | Assert | Got |
|---------|-------|--------|-----|
| Search TopK | `GPU_CUVS_CAGRA` | recall > 0.7 | **0.0** |
| Serialize/Deserialize | `GPU_CUVS_CAGRA` | recall ≥ 0.8 | **0.0** |
| Search TopK | `GPU_CUVS_IVF_PQ` | recall ≥ 0.85 | 0.837 (near-miss) |

Layer-2 gate `Test Gpu Index Search` (**108** asserts) stays **PASS**.

See also `docs/porting_milvus_gpu_to_amd.tex` §`sec:layer2-catch2-gaps`.

---

## Phase A — reproduce + triage

### Reproduce on lab

```bash
export WORKDIR=~/rocmds_check_gfx1100
cd ~/ann-harness-amd   # or clone of ann-harness-amd
git pull --ff-only

# hipVS Python (needed for library split). IVF microbench venv is fine if it has cagra:
source ~/hipvs-bench-venv/bin/activate   # create/build: docs/hipvs_vs_cuvs_bench.md §1
python3 -c "from cuvs.neighbors import cagra; print('cagra OK')"

bash scripts/reproduce_cagra_gfx1100.sh
# summary: $WORKDIR/logs/cagra_repro_summary_*.md

# Globs expanded inside classify (do not rely on bash alone):
python3 scripts/classify_cagra_triage.py \
  --hipvs-json "$WORKDIR/logs/lib_hipvs_cagra_minimal_*.json" \
  --catch2-log "$WORKDIR/logs/cagra_catch2_*.log"
# Or Catch2-only while Python cagra is missing:
# python3 scripts/classify_cagra_triage.py --catch2-log "$WORKDIR/logs/cagra_catch2_*.log"
```

Knowhere Catch2 on this tree does **not** accept `-k` (that token errors). The
repro script runs `"Test All GPU Index" -s` (same as Layer-2 docs).

### Triage ladder

```text
1. hipVS Python CAGRA on small SIFT (bypass Knowhere)
     source ~/hipvs-bench-venv/bin/activate
     MAX_TRAIN_ROWS=10000 MAX_QUERY_ROWS=200 \
       bash scripts/run_hipvs_cagra_bench.sh
   - recall ~0  → hipVS / RDNA3 kernel / wavefront issue → ROCm-DS / hipVS patch
   - recall OK  → Knowhere GPU_CUVS_CAGRA wiring / serialize → Knowhere HIP patch
   - import fails → build hipVS Python (docs/hipvs_vs_cuvs_bench.md §1); C++ libcuvs alone is not enough

2. Catch2 "Test All GPU Index" -s
   - TopK vs Serialize: if only Serialize fails, product load path is the B blocker

3. Capture: ROCm version, hipVS SHA, USE_WARPSIZE_32, BUILD_CAGRA_HNSWLIB,
   gfx target, full failing assert text
```

**Time-box:** if pure hipVS is dead on gfx1100 for >1 week with no consumer path,
deliver blocked + minimal repro to AMD; **do not** fake Phase B green.

### Lib microbench (after recall > 0)

```bash
# AMD
source ~/hipvs-bench-venv/bin/activate
bash scripts/run_hipvs_cagra_bench.sh

# NVIDIA peer
WORKDIR=~/milvus_cuda_4080 bash scripts/run_cuvs_cagra_bench.sh

python3 scripts/compare_cuvs_lib_json.py \
  --hipvs $WORKDIR/logs/lib_hipvs_cagra_….json \
  --cuvs  ~/milvus_cuda_4080/logs/lib_cuvs_cagra_….json
```

Freeze: `graph_degree=32`, `intermediate_graph_degree=64`, `itopk=32,64,128,256`,
`k=10`, SIFT-1M, metric `sqeuclidean`.

---

## Phase B — Milvus product (only after A)

```bash
# Milvus already up on :19530
bash scripts/run_milvus_gpu_cagra_smoke.sh

# Full SIFT
SKIP_START=1 bash scripts/run_milvus_layer4_cagra.sh

# Optional GIST (VRAM-heavy graph — may need smaller N or skip)
DATA_PATH=data/gist-960-euclidean.hdf5 \
  L4_COLLECTION=gist_gpu_cagra_l4 \
  SKIP_START=1 bash scripts/run_milvus_layer4_cagra.sh
```

Seal check: log must show `GPU_CUVS_CAGRA` / CAGRA activity, not CPU fallback /
`InvalidDeviceFunction`.

Fair CUDA peer: same `run_milvus_hdf5.py --index-type GPU_CAGRA` recipe on
RTX 4080 Docker/source Milvus v2.5.4.

---

## Dashboard / CI proposal (`amd-cagra`)

Suggest evidence rows (same style as IVF product / lib benches):

| Row id | Schema / artifact | Pass rule |
|--------|-------------------|-----------|
| `amd-cagra-lib` | `library_cuvs_api` JSON, index CAGRA | recall@10 ≥ threshold at itopk≥128; QPS recorded |
| `nvidia-cagra-lib` | same on CUDA | peer baseline |
| `cagra-lib` parity | ratio hip/cu QPS + recall Δ | recall Δ ≤ ε; QPS informational |
| `amd-cagra-smoke` | Milvus seal smoke log + client metrics | sealed GPU_CAGRA; recall > 0 |
| `amd-cagra-l4` / `nvidia-cagra-l4` | Layer-4 JSON | recall ladder vs itopk; fair peer |
| `cagra` product parity | AMD vs NVIDIA | matched params; gate on recall Δ |

Keep **HIP opt-in** (`MILVUS_WITH_HIP` / `WITH_HIP`) so NVIDIA CI never pulls ROCm.

Do **not** mark dashboard `amd-gtests` PASS until Catch2 CAGRA rows are green
(or dashboard filters to Layer-2 gate only).

---

## Scripts map

| Script | Role |
|--------|------|
| `reproduce_cagra_gfx1100.sh` | Phase A Catch2 + hipVS minimal |
| `bench_cuvs_cagra.py` | Lib API bench |
| `run_hipvs_cagra_bench.sh` / `run_cuvs_cagra_bench.sh` | Lab wrappers |
| `run_milvus_hdf5.py --index-type GPU_CAGRA` | Product client |
| `run_milvus_gpu_cagra_smoke.sh` | Phase B smoke |
| `run_milvus_layer4_cagra.sh` | Phase B Layer-4 |

---

## Status log

| Date | Item | Status |
|------|------|--------|
| 2026-07-26 | Catch2 CAGRA TopK/Serialize | FAIL recall 0.0 (baseline) |
| 2026-08-03 | Harness + docs landed in ann-harness | **ready** — run on `amd-rx7900xtx` |
| 2026-08-03 | Stakeholder one-pager + dashboard proposal | `cagra_stakeholder_onepager.md`, `cagra_dashboard_ci_proposal.md` |
| 2026-08-03 | Catch2 full GPU | 572/576; CAGRA recall **0.0** (×2); IVF_PQ near-misses |
| 2026-08-03 | hipVS `ivf_pq` build | **FAIL** `graph_core` invalid/duplicated neighbors |
| 2026-08-04 | GT bug (10k subset vs full HDF5 neighbors) | caused false 0.0695 on **both** GPUs |
| 2026-08-04 | hipVS + cuVS `nn_descent` 10k smoke (exact GT) | **recall@10 = 1.0** both; hip QPS ~0.33–0.41× cu |
| 2026-08-04 | **OWNER (lib quality)** | **cleared for nn_descent** on gfx1100 |
| (lab) | Full SIFT-1M lib + Catch2 (Knowhere may still use ivf_pq) | *next* |
| (lab) | Phase B smoke / L4 | *after Catch2 / Knowhere build-algo check* |

### First commands on the lab (copy-paste)

```bash
cd ~/ann-harness-amd && git pull --ff-only
export WORKDIR=~/rocmds_check_gfx1100
source ~/hipvs-bench-venv/bin/activate   # if missing, see hipvs_vs_cuvs_bench.md §1
bash scripts/reproduce_cagra_gfx1100.sh
python3 scripts/classify_cagra_triage.py \
  --hipvs-json "$WORKDIR/logs/lib_hipvs_cagra_minimal_*.json" \
  --catch2-log "$WORKDIR/logs/cagra_catch2_*.log"
```
