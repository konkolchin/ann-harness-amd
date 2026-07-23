# hipVS vs cuVS — library speed microbench

Goal: compare **hipVS** (AMD ROCm) and **cuVS** (NVIDIA CUDA) **directly** via the
shared Python `cuvs` API — **no Milvus / Knowhere**.

| Field | Freeze this |
|-------|-------------|
| Dataset | SIFT-1M `data/sift-128-euclidean.hdf5` |
| Client | `scripts/bench_cuvs_ivf.py` |
| `nlist` | 1024 |
| `k` | 10 |
| `nprobe` | 1,4,8,16,32 |
| Indexes | `IVF_FLAT`, then `IVF_PQ` with **`m=32`**, `nbits=8` |
| Metric | `sqeuclidean` (L2²; matches Milvus L2 ranking for recall) |

**Fairness:** same API + same recipe. GPUs differ (RX 7900 XTX vs RTX 4080) —
peer-class, not identical silicon. Report QPS **and** recall@10.

**Not the same as** Milvus Layer-4 HIP-vs-CUDA (product path). That includes
Knowhere + Milvus RPC. This microbench isolates the ANN library.

---

## 0) Shared data

```bash
cd ~/ann-harness-amd   # or your clone of konkolchin/ann-harness-amd
mkdir -p data
test -f data/sift-128-euclidean.hdf5 || \
  wget -c https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
    -O data/sift-128-euclidean.hdf5
```

Pull these scripts if needed:

```bash
git pull --ff-only origin master
# scripts/bench_cuvs_ivf.py
# scripts/run_hipvs_ivf_bench.sh
# scripts/run_cuvs_ivf_bench.sh
# scripts/compare_cuvs_lib_json.py
```

---

## 1) AMD — hipVS Python (RX 7900 XTX)

Layer‑1 already built **C++** `libcuvs` for Knowhere/Milvus. The library microbench needs
the **Python** package too (`import cuvs` — same name as NVIDIA). **No conda** — use `venv`.

### 1a) Venv + ROCm CuPy + deps

```bash
export WORKDIR=~/rocmds_check_gfx1100
export ROCM_HOME=/opt/rocm
export PATH="$ROCM_HOME/bin:$PATH"

# Prefer 3.11+ if available
python3 -m venv ~/hipvs-bench-venv
source ~/hipvs-bench-venv/bin/activate
pip install -U pip setuptools wheel
pip install numpy h5py cython scikit-build-core

# CuPy for ROCm (AMD-hosted wheel when available):
pip install amd-cupy --extra-index-url=https://pypi.amd.com/simple \
  || {
    echo "No amd-cupy wheel — building CuPy from source (slow)..."
    export CUPY_INSTALL_USE_HIP=1
    export HCC_AMDGPU_TARGET=gfx1100
    pip install cupy --no-cache-dir
  }

python3 -c "import cupy; print('cupy OK', cupy.__version__)"
```

### 1b) Build / install hipVS Python into that venv

C++ hipVS must already be installed (Layer‑1). Then:

```bash
source ~/hipvs-bench-venv/bin/activate
export WORKDIR=~/rocmds_check_gfx1100
export ROCM_HOME=/opt/rocm
export CMAKE_PREFIX_PATH="${WORKDIR}/install:${ROCM_HOME}:${CMAKE_PREFIX_PATH:-}"
# If your Layer-1 prefix differs, set it explicitly:
#   export CMAKE_PREFIX_PATH="$INSTALL_PREFIX:$ROCM_HOME"

cd "${WORKDIR}/hipVS"
# Pin discrete GPU only (host also has Raphael iGPU gfx1036 — breaks cmake arch detect)
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export AMDGPU_TARGETS=gfx1100
export CMAKE_HIP_ARCHITECTURES=gfx1100
# Drop stale CMake cache from the failed dual-arch configure:
rm -rf cpp/build CMakeCache.txt 2>/dev/null || true
# Rebuild python bindings against the installed libcuvs:
./build.sh libcuvs python

# If build.sh python target is awkward, install wheels manually:
#   cd "${WORKDIR}/hipVS/python/libcuvs" && pip install -v --no-build-isolation .
#   cd "${WORKDIR}/hipVS/python/cuvs"    && pip install -v --no-build-isolation .

python3 -c "import cuvs, cupy; print('ok', cuvs.__file__)"
python3 -c "from cuvs.neighbors import ivf_flat; print('neighbors OK')"
```

You may also need:

```bash
export LD_LIBRARY_PATH="${WORKDIR}/install/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"
```

### 1c) Run the library bench

```bash
cd ~/ann-harness-amd
source ~/hipvs-bench-venv/bin/activate
export WORKDIR=~/rocmds_check_gfx1100
export LD_LIBRARY_PATH="${WORKDIR}/install/lib:${ROCM_HOME:-/opt/rocm}/lib:${LD_LIBRARY_PATH:-}"

bash scripts/run_hipvs_ivf_bench.sh
INDEX_TYPE=IVF_PQ M=32 bash scripts/run_hipvs_ivf_bench.sh
```

JSON lands under `$WORKDIR/logs/lib_hipvs_*.json`.

Smoke (optional, small):

```bash
python3 scripts/bench_cuvs_ivf.py \
  --backend hipvs --index-type IVF_FLAT \
  --max-train-rows 10000 --max-query-rows 1000 \
  --data data/sift-128-euclidean.hdf5 \
  --results-json /tmp/hipvs_smoke.json
```

---

## 2) NVIDIA — cuVS Python (RTX 4080)

`milvus-bench-venv` (pymilvus only) does **not** include cuVS. Use a dedicated env.

### 2a) Check driver / Python

```bash
nvidia-smi | head -15          # note CUDA Version (12.x → use -cu12 wheels)
python3 --version              # RAPIDS cuVS wheels usually need Python >= 3.11
```

If `python3` is 3.10, create a 3.11+ venv (Ubuntu 22):

```bash
sudo apt-get install -y python3.11 python3.11-venv
python3.11 -m venv ~/cuvs-bench-venv
source ~/cuvs-bench-venv/bin/activate
pip install -U pip
```

### 2b) Install cuVS + CuPy (pip, CUDA 12) — recommended on this host

```bash
source ~/cuvs-bench-venv/bin/activate   # or reactivate
pip install -U pip
pip install "numpy" "h5py" \
  "cupy-cuda12x" \
  "cuvs-cu12" \
  --extra-index-url=https://pypi.nvidia.com
# Optional if CuPy later asks for CUDA headers:
#   pip install "cupy-cuda12x[ctk]"
# or: export CUDA_PATH=/usr/local/cuda

python3 -c "import cuvs, cupy; print('ok', getattr(cuvs,'__version__','?'), cupy.__version__)"
# neighbors needs libcuvs_c.so on LD_LIBRARY_PATH (pip wheels):
source ~/ann-harness-amd/scripts/cuvs_pip_ld_path.sh
python3 -c "from cuvs.neighbors import ivf_flat; print('neighbors OK')"
nvidia-smi -L
```

If your toolkit/driver reports **CUDA 13.x**, use `cuvs-cu13` and the matching CuPy CUDA 13 wheel instead.

### 2c) Run the library bench

```bash
cd ~/ann-harness-amd
source ~/cuvs-bench-venv/bin/activate    # must have import cuvs
export WORKDIR=~/milvus_cuda_4080
# wrappers source scripts/cuvs_pip_ld_path.sh automatically

bash scripts/run_cuvs_ivf_bench.sh
INDEX_TYPE=IVF_PQ M=32 bash scripts/run_cuvs_ivf_bench.sh
# or: bash scripts/run_lib_bench_both_indexes.sh
```

If you still see `libcuvs_c.so: cannot open shared object file`:

```bash
find ~/cuvs-bench-venv -name 'libcuvs_c.so*'
source ~/ann-harness-amd/scripts/cuvs_pip_ld_path.sh
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head
```

JSON under `$WORKDIR/logs/lib_cuvs_*.json`.

---

## 3) Compare

```bash
python3 scripts/compare_cuvs_lib_json.py \
  --hipvs ~/rocmds_check_gfx1100/logs/lib_hipvs_ivf_flat_YYYYMMDD_HHMMSS.json \
  --cuvs  ~/milvus_cuda_4080/logs/lib_cuvs_ivf_flat_YYYYMMDD_HHMMSS.json

python3 scripts/compare_cuvs_lib_json.py \
  --hipvs ~/rocmds_check_gfx1100/logs/lib_hipvs_ivf_pq_m32_YYYYMMDD_HHMMSS.json \
  --cuvs  ~/milvus_cuda_4080/logs/lib_cuvs_ivf_pq_m32_YYYYMMDD_HHMMSS.json
```

Speed-up = hipVS QPS / cuVS QPS.

---

## 4) How to read results

| Signal | Meaning |
|--------|---------|
| recall@10 match | Library search quality aligned |
| QPS ratio ~1× | Peer throughput on peer GPUs |
| Large QPS gap + matched recall | Hardware/stack speed, not wrong neighbors |
| recall diverge | Param / metric / build mismatch — fix before quoting speed |

PQ note: script maps Milvus **`m`** → cuVS **`pq_dim`**, **`nbits`** → **`pq_bits`**.

---

## 5) Interim (product path) until lab JSON lands

Milvus Layer-4 on the same hosts already showed **matched recall** and roughly
**~1.0–1.5×** AMD vs NVIDIA QPS for `GPU_IVF_FLAT` / `GPU_IVF_PQ`. That is
**hipVS under Knowhere+Milvus** vs **cuVS under Knowhere+Milvus**, not this
library microbench. Use product numbers only with that caveat.
