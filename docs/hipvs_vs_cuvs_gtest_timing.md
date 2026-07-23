# hipVS vs cuVS — shared unit-test timing (management ask)

**Manager question:** we got hipVS working on RDNA3 with `warp_size=32`. Is that
library competitive with NVIDIA cuVS?

**Answer method:** run the **same gtest suite both trees ship** (100+ cases), keep
correctness, and **record wall-clock per case**. No IVF_FLAT product story, no
Milvus, no custom SIFT recipe.

| Field | Freeze |
|-------|--------|
| Suite | Shared `cpp/build/gtests` binaries (default below) |
| Default IVF filter | `AnnIVFFlatTest/AnnIVFFlatTestF_float.*` (the 98/98 gfx1100 set) |
| Also timed | `NEIGHBORS_ANN_IVF_PQ_TEST`, `NEIGHBORS_ANN_BRUTE_FORCE_TEST` |
| Metric | Per-case ms from Google Test `(N ms)`; geo-mean / median of hip/cu |
| GPUs | Peer: RX 7900 XTX (hipVS) vs RTX 4080 (cuVS) — not same silicon |

**Caveat (one line for the manager):** these are correctness unit tests (small
synthetic sizes), not production QPS. They still exercise the same kernels both
libraries ship — best apples-to-apples “is the port in the ballpark?” signal.

The earlier Python SIFT microbench (`docs/hipvs_vs_cuvs_bench.md`) stays for
engineers; it is **not** the management brief for this ask.

---

## 0) Pull harness

```bash
cd ~/ann-harness-amd
git pull --ff-only origin master
```

Scripts:

- `scripts/time_cuvs_gtests.py` — run binaries, parse times → JSON  
- `scripts/run_hipvs_gtest_timing.sh` / `run_cuvs_gtest_timing.sh`  
- `scripts/compare_gtest_timings.py` — manager summary  

---

## 1) AMD — hipVS (RX 7900 XTX)

hipVS must be built **with tests** and `USE_WARPSIZE_32=ON` for gfx1100
(Layer 1 / 1.5). If `cpp/build/gtests` was cleaned, rebuild first.

### 1a) Locate or rebuild gtests

```bash
export WORKDIR=~/rocmds_check_gfx1100
export HIPVS_ROOT=$WORKDIR/hipVS
export INSTALL_PREFIX=$WORKDIR/install
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0

# Already built?
find "$WORKDIR" -name NEIGHBORS_ANN_IVF_FLAT_TEST -type f 2>/dev/null | head

# If empty — rebuild float IVF suite only (~98 cases; enough for manager ask).
# --gpu-arch MUST be last. Nested quotes on --cmake-args (bash trap).
cd "$HIPVS_ROOT"
INSTALL_PREFIX=$INSTALL_PREFIX ./build.sh libcuvs tests \
  '--cmake-args="-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF"' \
  --gpu-arch="gfx1100" \
  --limit-tests=NEIGHBORS_ANN_IVF_FLAT_TEST

ls -la "$HIPVS_ROOT/cpp/build/gtests/NEIGHBORS_ANN_IVF_FLAT_TEST"
```

### 1b) Time them

```bash
cd ~/ann-harness-amd
GTEST_BINARIES=NEIGHBORS_ANN_IVF_FLAT_TEST \
  bash scripts/run_hipvs_gtest_timing.sh
# JSON → $WORKDIR/logs/gtest_timing_hipvs_*.json
```

If the binary lives elsewhere:

```bash
GTEST_DIR=/path/to/gtests GTEST_BINARIES=NEIGHBORS_ANN_IVF_FLAT_TEST \
  bash scripts/run_hipvs_gtest_timing.sh
```

Full binaries (no float-only filter):

```bash
NO_DEFAULT_FILTERS=1 bash scripts/run_hipvs_gtest_timing.sh
```

---

## 2) NVIDIA — cuVS (RTX 4080)

Pip/`cuvs-cu12` does **not** ship gtest binaries. You need a **source** build with
`tests` (same as AMD), or an existing `NEIGHBORS_ANN_IVF_FLAT_TEST`.

```bash
export WORKDIR=~/milvus_cuda_4080
export CUDA_VISIBLE_DEVICES=0

# Already built somewhere?
find "$WORKDIR" "$HOME" -name NEIGHBORS_ANN_IVF_FLAT_TEST -type f 2>/dev/null | head

# If empty — clone + build float IVF suite only:
mkdir -p "$WORKDIR" && cd "$WORKDIR"
test -d cuvs || git clone --depth 1 --branch branch-25.02 https://github.com/rapidsai/cuvs.git
cd cuvs
# Prefer a RAPIDS-compatible env if you have one; otherwise system CUDA + cmake.
./build.sh libcuvs tests \
  --limit-tests=NEIGHBORS_ANN_IVF_FLAT_TEST \
  --gpu-arch="89-real"

ls -la "$(pwd)/cpp/build/gtests/NEIGHBORS_ANN_IVF_FLAT_TEST"

cd ~/ann-harness-amd
GTEST_DIR=$WORKDIR/cuvs/cpp/build/gtests \
  GTEST_BINARIES=NEIGHBORS_ANN_IVF_FLAT_TEST \
  bash scripts/run_cuvs_gtest_timing.sh
```

Use the **same** float filter as AMD (script default). No `USE_WARPSIZE_32` on CUDA.

---

## 3) Compare (either box / laptop)

```bash
python3 scripts/compare_gtest_timings.py \
  --hipvs /path/to/gtest_timing_hipvs_....json \
  --cuvs  /path/to/gtest_timing_cuvs_....json \
  --out-json results/gtest_timing/COMPARE.json \
  --out-md   results/gtest_timing/COMPARE.md
```

**Manager line** is printed automatically from geo-mean ratio
(`hip_ms / cu_ms`):

| Geo-mean | How to say it |
|----------|----------------|
| ≲ 1.25× | roughly competitive |
| ~1.25–2× | same ballpark; cuVS somewhat faster |
| ≫ 2× | not competitive on wall time yet |

Plus: pass counts must both be green (or explain failures).

---

## 4) Copy results into the repo

```bash
mkdir -p results/gtest_timing
cp "$WORKDIR/logs"/gtest_timing_*.json results/gtest_timing/
```

Then re-run compare and paste the manager line into slides.
