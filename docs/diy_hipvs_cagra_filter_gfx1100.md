# DIY: hipVS CAGRA filtered search on gfx1100

**Goal:** Make `cuvs.neighbors.cagra.search(..., filter=filters.from_bitset(...))`
green on RX 7900 XTX / `gfx1100`, using your fork
[konkolchin/hipVS](https://github.com/konkolchin/hipVS).

**Gate (must stay green):**

```bash
bash scripts/run_hipvs_cagra_filter_repro.sh
# unfiltered R@1 ≈ 1.0
# filter_40pct R@1 > 0.7  (today: 0.0)
# simple_bitset_64 R@1 > 0.8, neg1 ≈ 0  (today: all -1)
```

**Out of scope for week‑1:** Knowhere-only patches, TensorRT/FA3, Instinct-only work.

Evidence already locked: `docs/escalation_hipvs_cagra_filter_gfx1100.md`,
`$WORKDIR/logs/lib_hipvs_cagra_filter_20260805_232922.json`.

---

## 0) Match lab install to a git SHA

On `amd-rx7900xtx`:

```bash
export WORKDIR=~/rocmds_check_gfx1100
export ROCM_HOME=/opt/rocm

# Where is the tree that produced install/?
ls -la "$WORKDIR/hipVS" "$WORKDIR/install" 2>/dev/null | head

cd "$WORKDIR/hipVS" 2>/dev/null || {
  echo "No $WORKDIR/hipVS — clone fork in §1"
  exit 0
}

echo "=== local hipVS ==="
git remote -v
git log -1 --oneline
git rev-parse HEAD
git describe --tags --always 2>/dev/null || true

# Installed libs (provenance clues)
ls -lt "$WORKDIR/install/lib"/libcuvs* 2>/dev/null | head
strings "$WORKDIR/install/lib/libcuvs.so" 2>/dev/null \
  | grep -E 'amdgcn-amd-amdhsa--gfx[0-9]+' | sort -u

# Python package in use by the failing repro
source ~/hipvs-bench-venv/bin/activate
python3 - <<'PY'
import cuvs
print("cuvs_file", cuvs.__file__)
print("version", getattr(cuvs, "__version__", "?"))
PY
```

**Decision:**

| Lab state | DIY base |
|-----------|----------|
| `$WORKDIR/hipVS` SHA known | Branch DIY off **that SHA** first (bisect apples-to-apples) |
| Only `install/`, no tree | Clone fork, prefer `release/rocmds-25.10` if lab was built from that (Layer‑1 docs); else try `release/rocmds-26.03` and re-validate unfiltered first |
| Fork stale vs upstream | `git fetch upstream` — do **not** jump to 26.03 mid-fix unless unfiltered still passes |

Lab Layer‑1 historically used **`release/rocmds-25.10`** + ROCm ≥ 7.0.2.

---

## 1) Sync your fork

Fork exists: `https://github.com/konkolchin/hipVS`  
Parent: `AMD-Ecosystem/hipVS` (ROCm-DS line). Default on fork is still `release/rocmds-25.10` (stale vs parent `26.03`).

```bash
export WORKDIR=~/rocmds_check_gfx1100
cd "$WORKDIR"

# If you already have a tree, reuse it; else:
if [ ! -d hipVS/.git ]; then
  git clone https://github.com/konkolchin/hipVS.git hipVS
fi
cd hipVS

git remote -v
# expect origin -> konkolchin/hipVS
git remote add upstream https://github.com/AMD-Ecosystem/hipVS.git 2>/dev/null || true
git fetch origin
git fetch upstream

# Option A — match old lab (recommended start)
git checkout -B fix/cagra-filter-gfx1100 origin/release/rocmds-25.10

# Option B — if lab already on 26.03 / you want latest
# git checkout -B fix/cagra-filter-gfx1100 upstream/release/rocmds-26.03

# If §0 found a precise SHA:
# git checkout -B fix/cagra-filter-gfx1100 <LAB_SHA>
```

Push the branch to your fork when ready:

```bash
git push -u origin fix/cagra-filter-gfx1100
```

---

## 2) Rebuild C++ + Python (gfx1100 pins)

Same discipline as Layer‑1 / `docs/hipvs_vs_cuvs_bench.md` §1b:

```bash
export WORKDIR=~/rocmds_check_gfx1100
export ROCM_HOME=/opt/rocm
export INSTALL_PREFIX="${INSTALL_PREFIX:-$WORKDIR/install}"
export PATH="$ROCM_HOME/llvm/bin:$ROCM_HOME/bin:$PATH"
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export AMDGPU_TARGETS=gfx1100
export CMAKE_HIP_ARCHITECTURES=gfx1100
export GPU_TARGETS=gfx1100
export CMAKE_ARGS="-DCMAKE_HIP_ARCHITECTURES=gfx1100 -DAMDGPU_TARGETS=gfx1100 -DGPU_TARGETS=gfx1100"
export SKBUILD_CMAKE_ARGS="${CMAKE_ARGS}"
export CMAKE_PREFIX_PATH="${INSTALL_PREFIX}:${ROCM_HOME}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"

cd "$WORKDIR/hipVS"
./build.sh clean 2>/dev/null || true
rm -rf cpp/build python/libcuvs/build python/cuvs/build 2>/dev/null || true

# Critical for RDNA3 wavefront-32 (same as IVF packer campaign):
./build.sh libcuvs --gpu-arch="gfx1100" \
  --cmake-args='-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF'

# Install prefix (adjust if your build.sh uses -DCMAKE_INSTALL_PREFIX differently)
cmake --install cpp/build --prefix "$INSTALL_PREFIX" 2>/dev/null \
  || ninja -C cpp/build install

# Python into the venv that runs the filter repro
source ~/hipvs-bench-venv/bin/activate
./build.sh libcuvs python --gpu-arch="gfx1100" \
  --cmake-args='-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF' \
  || {
    cd python/libcuvs && pip install -v --no-build-isolation --no-cache-dir .
    cd ../cuvs && pip install -v --no-build-isolation --no-cache-dir .
  }

cd ~
python3 -c "import cuvs; from cuvs.neighbors import cagra, filters; print(cuvs.__file__)"
```

**Smoke after rebuild (must not regress):**

```bash
cd ~/ann-harness-amd && git pull --ff-only
source ~/hipvs-bench-venv/bin/activate
export WORKDIR=~/rocmds_check_gfx1100
bash scripts/run_hipvs_cagra_filter_repro.sh
# Expect: unfiltered still 1.0; filter still red until you patch
```

---

## 3) Triage ladder (1–2 days)

Work **only** with `probe_cuvs_cagra_filter.py` + optional C++ gtest. Do not touch Knowhere yet.

### 3a) Confirm filter object / polarity

Already using `filters.from_bitset` (cuVS: bit **1** = allowed). Unfiltered OK ⇒ build/search OK.

### 3b) Code search in hipVS tree

```bash
cd "$WORKDIR/hipVS"
rg -n 'bitset_filter|from_bitset|filtering_rate|sample_filter' \
  cpp/include cpp/src python -g '*.{hpp,cuh,h,py,pyx}' | head -80

rg -n 'cagra.*search|search_plan|single_cta|multi_cta' \
  cpp/include/cuvs/neighbors/cagra* cpp/src/neighbors/cagra 2>/dev/null | head -40
```

Hot spots (names vary by branch):

- `cpp/src/neighbors/cagra/*search*`
- `cpp/include/cuvs/neighbors/common.hpp` / filtering
- Any `bitset` / `ballot` / `lane_mask` / `warp_size` in the search path

### 3c) Hypotheses (ordered)

1. **Wavefront 32:** filter mask / ballot assumes 64 (Instinct). Check `USE_WARPSIZE_32` actually applied in the search TU (`ninja -t query` / compile DB).
2. **Bitset endian / word packing:** Python packs uint32 little-endian bit‑i → sample i; device may read differently.
3. **Filter ignored:** wrong IDs with `neg1=0` on 40% case looks like “search OK, mask not applied” or “wrong allowed set”.
4. **Over-filter:** simple_bitset all `-1` looks like “everything excluded” or empty candidate set.

### 3d) Minimal C++ repro (optional)

If Python bindings obscure the bug, add/run a tiny C++ test that:

- builds CAGRA `nn_descent` on 64–10k vectors  
- searches with `bitset_filter`  
- prints first ids  

Prefer existing hipVS/cuvs filter tests if present:

```bash
cd "$WORKDIR/hipVS"
rg -n 'bitset|filter' cpp/tests -g '*cagra*' | head -40
# run the matching gtest target once identified
```

---

## 4) Fix loop

1. Patch hipVS (prefer smallest change in filter + CAGRA search).  
2. Rebuild `libcuvs` + Python (`§2`).  
3. Re-run `run_hipvs_cagra_filter_repro.sh`.  
4. Commit on `fix/cagra-filter-gfx1100` with repro JSON before/after.  
5. When green: Knowhere Catch2 bitset sections (optional confirmation).

```bash
cd ~/ann-harness-amd
export WORKDIR=~/rocmds_check_gfx1100
# after hipVS fix + knowhere still linked to new libcuvs:
bash scripts/apply_knowhere_cagra_nn_descent_ut.sh   # if not already
# rebuild knowhere_tests if it statically embeds; usually dynamic libcuvs
KT=$WORKDIR/knowhere/build/tests/ut/knowhere_tests
"$KT" "Test All GPU Index" -s 2>&1 | tee "$WORKDIR/logs/cagra_catch2_after_hipvs_filter.log"
grep -nE 'FAILED:|GPU_CUVS_CAGRA' "$WORKDIR/logs/cagra_catch2_after_hipvs_filter.log" | head
```

---

## 5) Upstream

When filter repro is green on gfx1100:

1. Push `fix/cagra-filter-gfx1100` to `konkolchin/hipVS`.  
2. Open PR → `AMD-Ecosystem/hipVS` (or ROCm-DS mirror) against the branch you based on.  
3. PR body: link harness repro + before/after JSON + note consumer `gfx1100` / `USE_WARPSIZE_32`.  
4. Keep filing/updating the escalation issue with the PR URL.

---

## 6) Time-box / go–no-go

| Day | Exit |
|-----|------|
| 0–1 | Fork synced, rebuild matches lab, unfiltered still 1.0 |
| 2–3 | Hypothesis narrowed (warp vs pack vs ignore) with evidence |
| 5–10 | Patch green on filter repro **or** escalate with failing kernel + notes |
| After green | Catch2 bitset + optional Milvus filtered smoke |

If after **~1 week** you only have “Instinct kernels assume wf64” without a fix path, attach findings to the ROCm-DS issue and pause DIY.

---

## Scripts map

| Artifact | Role |
|----------|------|
| `scripts/probe_cuvs_cagra_filter.py` | Ownership gate |
| `scripts/run_hipvs_cagra_filter_repro.sh` | Lab wrapper |
| `docs/escalation_hipvs_cagra_filter_gfx1100.md` | Bug report template |
| `scripts/collect_hipvs_cagra_filter_escalation.sh` | Env bundle |
| This doc | DIY runbook |

---

## First commands (copy-paste)

```bash
# amd@amd-rx7900xtx
cd ~/ann-harness-amd && git pull --ff-only
export WORKDIR=~/rocmds_check_gfx1100
source ~/hipvs-bench-venv/bin/activate

# §0 provenance
ls -la "$WORKDIR/hipVS" 2>/dev/null
(cd "$WORKDIR/hipVS" 2>/dev/null && git log -1 --oneline && git rev-parse HEAD)

# §1 fork branch (if tree missing, clone first)
# git clone https://github.com/konkolchin/hipVS.git $WORKDIR/hipVS-diy
```
