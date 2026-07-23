#!/usr/bin/env bash
# Library-level NVIDIA cuVS (cuvs Python API) IVF bench on CUDA.
#
# Prerequisites: RAPIDS/pip cuVS + CuPy CUDA, SIFT HDF5.
# See docs/hipvs_vs_cuvs_bench.md
#
# Usage:
#   bash scripts/run_cuvs_ivf_bench.sh
#   INDEX_TYPE=IVF_PQ M=32 WORKDIR=~/milvus_cuda_4080 bash scripts/run_cuvs_ivf_bench.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/milvus_cuda_4080}"
INDEX_TYPE="${INDEX_TYPE:-IVF_FLAT}"
M="${M:-32}"
NBITS="${NBITS:-8}"
NLIST="${NLIST:-1024}"
NPROBES="${NPROBES:-1,4,8,16,32}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="cuvs_${INDEX_TYPE,,}_m${M}"
if [ "${INDEX_TYPE}" = "IVF_FLAT" ]; then
  TAG="cuvs_ivf_flat"
fi
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/lib_${TAG}_${TS}.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# pip wheels put libcuvs_c.so under site-packages; dlopen needs LD_LIBRARY_PATH
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/cuvs_pip_ld_path.sh"

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: missing ${DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
echo "==> cuVS library bench"
echo "    index=${INDEX_TYPE} nlist=${NLIST} m=${M} nbits=${NBITS}"
echo "    results=${RESULTS_JSON}"

cd "${REPO_ROOT}"
python3 -c "from cuvs.neighbors import ivf_flat; import cupy; print('cuvs neighbors OK', cupy.__version__)"

EXTRA=()
if [ "${INDEX_TYPE}" = "IVF_PQ" ]; then
  EXTRA+=(--m "${M}" --nbits "${NBITS}")
fi

python3 scripts/bench_cuvs_ivf.py \
  --backend cuvs \
  --index-type "${INDEX_TYPE}" \
  --nlist "${NLIST}" \
  --nprobes "${NPROBES}" \
  --data "${DATA_PATH}" \
  --results-json "${RESULTS_JSON}" \
  "${EXTRA[@]}"

echo ""
echo "CUVS LIB BENCH OK"
echo "  results: ${RESULTS_JSON}"
