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
# origin MUST be https://github.com/konkolchin/hipVS.git (your fork).
# Lab trees often have origin -> ROCm-DS/hipVS (read-only for you) — fix:

if git remote get-url origin 2>/dev/null | grep -qE 'ROCm-DS/hipVS|AMD-Ecosystem/hipVS'; then
  git remote rename origin upstream_rocmds 2>/dev/null \
    || git remote remove origin
  git remote add origin https://github.com/konkolchin/hipVS.git
fi
git remote add upstream https://github.com/AMD-Ecosystem/hipVS.git 2>/dev/null || true
# optional: keep ROCm-DS as a named remote
# git remote add rocmds https://github.com/ROCm-DS/hipVS.git 2>/dev/null || true

git fetch origin
git fetch upstream

# Option A — match old lab (recommended start)
git checkout -B fix/cagra-filter-gfx1100 origin/release/rocmds-25.10

# Option B — if lab already on 26.03 / you want latest
# git checkout -B fix/cagra-filter-gfx1100 upstream/release/rocmds-26.03

# If §0 found a precise SHA (preferred when rebuilding against known-good install):
# git checkout -B fix/cagra-filter-gfx1100 87877f15fe6fdd6525730385e9474117ade6ecb3
```

Push the branch to **your fork** (`origin` = `konkolchin/hipVS`).  
GitHub **rejects account passwords** for `git push`; use a [PAT](https://github.com/settings/tokens) (HTTPS, paste as password) or SSH.

```bash
git rev-parse HEAD   # expect 87877f15… if you pinned that SHA
git remote -v        # origin -> konkolchin/hipVS
git push -u origin fix/cagra-filter-gfx1100
# Username: konkolchin
# Password: <classic PAT with repo scope, not your GitHub login password>
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

# Python into the venv that runs the filter repro.
# Note: CMAKE_ARGS / SKBUILD with HIP arch often poisons host g++ on
# python/cuvs (amd-hipvs) with `--offload-arch=gfx1100` → see §2b.
source ~/hipvs-bench-venv/bin/activate
./build.sh libcuvs python --gpu-arch="gfx1100" \
  --cmake-args='-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF' \
  || true

# Prefer finishing wheels explicitly (§2b) if build.sh python fails.
```

### 2b) Fix `amd-hipvs`: `g++: unrecognized … --offload-arch=gfx1100`

`python/cuvs` compiles Cython **host** `.cxx` with `g++`. HIP arch flags from
`CMAKE_ARGS` / `SKBUILD_CMAKE_ARGS` must not reach that compiler.

Filter kernels live in **C++ `libcuvs`**. If `cmake --install` / `amd-libhipvs`
already picked up the new `.so`, you can skip rebuilding `amd-hipvs` for
kernel experiments and just re-run the repro. Rebuild `amd-hipvs` when the
Python API/bindings must match a header change.

```bash
source ~/hipvs-bench-venv/bin/activate
export WORKDIR=~/rocmds_check_gfx1100
export ROCM_HOME=/opt/rocm
export INSTALL_PREFIX="${INSTALL_PREFIX:-$WORKDIR/install}"
export PATH="$ROCM_HOME/llvm/bin:$ROCM_HOME/bin:$PATH"
export CMAKE_PREFIX_PATH="${INSTALL_PREFIX}:${ROCM_HOME}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0

# Keep device pin; drop skbuild HIP args that leak onto host CXX
unset CMAKE_ARGS SKBUILD_CMAKE_ARGS

# Belt-and-suspenders: strip HIP flags if CMake still injects them
cat > /tmp/gxx-strip-offload <<'EOF'
#!/bin/bash
args=()
for a in "$@"; do
  case "$a" in
    --offload-arch=*|-offload-arch=*|--cuda-gpu-arch=*|-xhip|hip) continue ;;
  esac
  args+=("$a")
done
exec /usr/bin/x86_64-linux-gnu-g++ "${args[@]}"
EOF
chmod +x /tmp/gxx-strip-offload
export CXX=/tmp/gxx-strip-offload
export CMAKE_CXX_COMPILER=/tmp/gxx-strip-offload

# Low-level wheel (bundles/links libcuvs) — keep USE_WARPSIZE_32 for C++ rebuilds
cd "$WORKDIR/hipVS/python/libcuvs"
pip install -v --no-build-isolation --no-cache-dir . \
  || echo "amd-libhipvs already OK — continue"

# High-level package (host Cython only)
cd "$WORKDIR/hipVS/python/cuvs"
rm -rf build
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple \
  "hip-python-as-cuda" 2>/dev/null || true
pip install -v --no-build-isolation --no-cache-dir .

# Must not be inside hipVS/python/cuvs (source tree shadows the wheel).
cd ~
python3 -c "import cuvs; from cuvs.neighbors import cagra, filters; print(cuvs.__file__)"
# expect: …/site-packages/cuvs/__init__.py
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

1. **✅ Applied (26.03 backport):** `move_invalid_to_end_of_list` uint32
   `mask_type` when `warp_size()==32` (§4a). Changed simple_bitset from
   all `-1` → wrong IDs; did **not** fix recall.
2. **✅ Ruled out — bitset packing / `test()`:** `filter_all_ones` R@1=1.0 and
   `brute_force+allow_only_0` R@1=1.0 (same bits). Polarity OK.
3. **✅ OWNER — CAGRA filtered graph walk** (`SEARCH_ALGO=single_cta`):
   selective bitset → wrong allowed IDs / empty. Next: remaining
   `bitmask_type` + `__ballot_sync` sites in
   `search_single_cta_kernel-inl.cuh` (e.g. `pickup_next_parents`
   `(1 << threadIdx.x)` ranking), filter post-process `WarpScan` /
   `shfl_xor`, and diff that file vs `AMD-Ecosystem/hipVS`
   `release/rocmds-26.03`.
4. multi-CTA still needs its own audit (`SEARCH_ALGO=multi_cta`) after
   single_cta is green.

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

### 4a) First patch — backport 26.03 `move_invalid` mask width

File: `cpp/src/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh`  
Function: `move_invalid_to_end_of_list` (only used from the **filtered** search path).

Replace the ballot block with the 26.03 version:

```cpp
        // Check if the index is invalid
        const auto I_found_invalid = (index == invalid_index);
        using mask_type = std::conditional_t<raft::warp_size() == 32, uint32_t, uint64_t>;
        // We're intentionally not using bitmask_type(uint64_t) here.
        // Note the expression `(who_has_invalid << (raft::warp_size() - lane_id)` which is trying
        // to compute the following: It is trying to check if there is any smaller lane that has
        // found invalid index. By left shifting the bitmask by (warp_size - lane_id), All the lanes
        // that found an invalid index will have it's bit shifted out except for the lanes with
        // smaller lane_id than the current lane. For this to work correctly, the mask_type should
        // not be wider than the actual warp size.
        const mask_type who_has_invalid = raft::ballot(I_found_invalid, __activemask());
        // if a value that is loaded by a smaller lane id thread, shift the array
        if ((who_has_invalid << (raft::warp_size() - lane_id)) and i > 0) {
          index_array[i - 1]    = index;
          distance_array[i - 1] = distance;
        }

        found_invalid = who_has_invalid;
```

(Add `#include <type_traits>` / `<cstdint>` near the top of the `.cuh` if the TU does not already see them.)

**Status 2026-08-10:** §4a applied on lab; gate still red on selective filters.
Bitset OK (brute_force). Continue with §4b.

### 4b) Next — other wf32 ballot sites in single-CTA CAGRA

```bash
cd "$WORKDIR/hipVS"
# Force single_cta in the harness gate:
SEARCH_ALGO=single_cta bash ~/ann-harness-amd/scripts/run_hipvs_cagra_filter_repro.sh

rg -n 'bitmask_type|__ballot_sync|raft::ballot|1 << threadIdx' \
  cpp/src/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh

# Diff filter-related hunks vs upstream fix branch:
curl -sL \
  https://raw.githubusercontent.com/AMD-Ecosystem/hipVS/release/rocmds-26.03/cpp/src/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh \
  > /tmp/cagra_single_2603.cuh
diff -u cpp/src/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh \
  /tmp/cagra_single_2603.cuh | head -200
```

**§4b.1 — `pickup_next_parents` (remaining `bitmask_type` + `1 << threadIdx.x`):**

```bash
bash ~/ann-harness-amd/scripts/apply_hipvs_cagra_pickup_parents_wf32.sh "$WORKDIR/hipVS"
# then rebuild libcuvs + amd-libhipvs (§2 / §2b) and re-gate SEARCH_ALGO=single_cta
```

This (1) restores first-warp-only (upstream had `// if (threadIdx.x >= 32) return`
commented out), (2) ranks with `lane_id`, (3) uses `uint32_t` ballot mask on wf32.

Then:

1. Rebuild `libcuvs` (+ `python/libcuvs` wheel) — `§2` / `§2b`.  
2. `SEARCH_ALGO=single_cta bash scripts/run_hipvs_cagra_filter_repro.sh`  
3. Commit on `fix/cagra-filter-gfx1100` with before/after JSON.  
4. When green: Knowhere Catch2 bitset sections (optional confirmation).

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
