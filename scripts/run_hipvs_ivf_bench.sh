#!/usr/bin/env bash
# Library-level hipVS (cuvs Python API) IVF bench on AMD ROCm.
#
# Prerequisites: hipVS Python build (import cuvs), CuPy ROCm, SIFT HDF5.
# See docs/hipvs_vs_cuvs_bench.md  (venv: ~/hipvs-bench-venv recommended)
#
# Usage:
#   source ~/hipvs-bench-venv/bin/activate
#   bash scripts/run_hipvs_ivf_bench.sh
#   INDEX_TYPE=IVF_PQ M=32 bash scripts/run_hipvs_ivf_bench.sh
#   INDEX_TYPE=IVF_PQ M=32 LUT_DTYPE=float16 bash scripts/run_hipvs_ivf_bench.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
INDEX_TYPE="${INDEX_TYPE:-IVF_FLAT}"
M="${M:-32}"
NBITS="${NBITS:-8}"
LUT_DTYPE="${LUT_DTYPE:-float32}"
INTERNAL_DISTANCE_DTYPE="${INTERNAL_DISTANCE_DTYPE:-float32}"
NLIST="${NLIST:-1024}"
NPROBES="${NPROBES:-1,4,8,16,32}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="hipvs_${INDEX_TYPE,,}_m${M}"
if [ "${INDEX_TYPE}" = "IVF_FLAT" ]; then
  TAG="hipvs_ivf_flat"
elif [ "${INDEX_TYPE}" = "IVF_PQ" ] && [ "${LUT_DTYPE}" != "float32" ]; then
  TAG="hipvs_ivf_pq_m${M}_${LUT_DTYPE}"
fi
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/lib_${TAG}_${TS}.json}"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCM_HOME
export LD_LIBRARY_PATH="${WORKDIR}/install/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"

if ! python3 -c "from cuvs.neighbors import ivf_flat" 2>/dev/null; then
  echo "ERROR: Python cannot import cuvs.neighbors (hipVS Python not installed)." >&2
  echo "  Create ~/hipvs-bench-venv and build hipVS python — see docs/hipvs_vs_cuvs_bench.md §1" >&2
  echo "  Quick check: python3 -c \"from cuvs.neighbors import ivf_flat\"" >&2
  exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: missing ${DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
echo "==> hipVS library bench"
echo "    index=${INDEX_TYPE} nlist=${NLIST} m=${M} nbits=${NBITS}"
if [ "${INDEX_TYPE}" = "IVF_PQ" ]; then
  echo "    lut_dtype=${LUT_DTYPE} internal_distance_dtype=${INTERNAL_DISTANCE_DTYPE}"
fi
echo "    results=${RESULTS_JSON}"

cd "${REPO_ROOT}"
python3 -c "from cuvs.neighbors import ivf_flat; import cupy; print('cuvs neighbors OK', cupy.__version__)"

EXTRA=()
if [ "${INDEX_TYPE}" = "IVF_PQ" ]; then
  EXTRA+=(--m "${M}" --nbits "${NBITS}"
          --lut-dtype "${LUT_DTYPE}"
          --internal-distance-dtype "${INTERNAL_DISTANCE_DTYPE}")
fi
# hipVS on gfx1100 can hit hipErrorInvalidValue on n_queries=1 (p99 path); skip by default
P99_SAMPLE="${P99_SAMPLE:-0}"

python3 scripts/bench_cuvs_ivf.py \
  --backend hipvs \
  --index-type "${INDEX_TYPE}" \
  --nlist "${NLIST}" \
  --nprobes "${NPROBES}" \
  --p99-sample "${P99_SAMPLE}" \
  --data "${DATA_PATH}" \
  --results-json "${RESULTS_JSON}" \
  "${EXTRA[@]}"

echo ""
echo "HIPVS LIB BENCH OK"
echo "  results: ${RESULTS_JSON}"
