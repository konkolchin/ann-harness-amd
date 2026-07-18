#!/usr/bin/env bash
# Fair CPU-vs-GPU compare: VectorDBBench SIFT-1M IVF nprobe sweep.
#
# MODE=cpu  -> vectordbbench milvusivfflat   (Docker CPU Milvus IVF_FLAT)
# MODE=gpu  -> vectordbbench milvusgpuivfflat (HIP/CUDA GPU Milvus GPU_IVF_FLAT)
#
# Same recipe as docs/ann_framework_runbook.tex § vectordbbench-ivf-nprobe:
#   PerformanceCustomDataset, nlist=1024, k=10, nprobe=1,4,8,16,32
#   load once on first nprobe; --skip-load for the rest
#
# Prerequisites:
#   - source ~/vdbbench-venv/bin/activate  (Python >= 3.11, vectordb-bench)
#   - ~/vdbbench-sift1m/{train,test,neighbors}.parquet
#   - Milvus healthy on URI (CPU Docker or HIP standalone — not both)
#
# Usage:
#   MODE=cpu bash scripts/run_vdbbench_milvus_ivf_sweep.sh
#   MODE=gpu bash scripts/run_vdbbench_milvus_ivf_sweep.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-cpu}"
URI="${MILVUS_URI:-http://127.0.0.1:19530}"
LISTS="${LISTS:-1024}"
K="${K:-10}"
NPROBES="${NPROBES:-1,4,8,16,32}"
DATASET_DIR="${DATASET_DIR:-${HOME}/vdbbench-sift1m}"
CASE_NAME="${CASE_NAME:-SIFT1M-IVF}"
DATASET_NAME="${DATASET_NAME:-SIFT1M}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
TS="$(date -u +%Y%m%d_%H%M%S)"
DB_LABEL="${DB_LABEL:-amd-rx7900xtx-${MODE}}"
VENV="${VDBBENCH_VENV:-${HOME}/vdbbench-venv}"
# Required by vectordbbench milvusgpuivfflat (string/float click options).
CACHE_ON_DEVICE="${CACHE_ON_DEVICE:-false}"
REFINE_RATIO="${REFINE_RATIO:-1.0}"

case "${MODE}" in
  cpu) CMD=milvusivfflat ;;
  gpu) CMD=milvusgpuivfflat ;;
  *)
    echo "ERROR: MODE must be cpu or gpu (got: ${MODE})" >&2
    exit 1
    ;;
esac

LOG="${LOG:-${LOG_DIR}/vdb_${MODE}_ivf_nprobe_${TS}.log}"
mkdir -p "${LOG_DIR}"

if [ -f "${VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

if ! command -v vectordbbench >/dev/null 2>&1; then
  echo "ERROR: vectordbbench not on PATH. Activate venv first:" >&2
  echo "  source ~/vdbbench-venv/bin/activate" >&2
  exit 1
fi

for f in train.parquet test.parquet neighbors.parquet; do
  if [ ! -f "${DATASET_DIR}/${f}" ]; then
    echo "ERROR: missing ${DATASET_DIR}/${f}" >&2
    echo "  See docs/vdbbench_cpu_gpu_compare.md (parquet prep) or runbook § vectordbbench-ivf-nprobe" >&2
    exit 1
  fi
done

if ! curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1 \
  && ! (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
  echo "ERROR: Milvus not reachable on ${URI} (:19530 / :9091)" >&2
  if [ "${MODE}" = "cpu" ]; then
    echo "  Start Docker CPU Milvus (milvus-docker / compose) first." >&2
  else
    echo "  Start HIP Milvus with ROCR_VISIBLE_DEVICES=0 first." >&2
  fi
  exit 1
fi

echo "==> MODE=${MODE} CMD=${CMD}"
echo "==> URI=${URI} lists=${LISTS} k=${K} nprobes=${NPROBES}"
echo "==> dataset=${DATASET_DIR}"
echo "==> log=${LOG}"
echo "==> vectordbbench: $(command -v vectordbbench)"
vectordbbench "${CMD}" --help >/dev/null

IFS=',' read -r -a PROBE_ARR <<< "${NPROBES}"
FIRST=1
for NPROBE in "${PROBE_ARR[@]}"; do
  NPROBE="$(echo "${NPROBE}" | tr -d '[:space:]')"
  [ -n "${NPROBE}" ] || continue
  echo "======== nprobe=${NPROBE} ========" | tee -a "${LOG}"
  if [ "${FIRST}" = "1" ]; then
    EXTRA=(--drop-old --load)
    FIRST=0
  else
    EXTRA=(--skip-drop-old --skip-load)
  fi
  GPU_EXTRA=()
  if [ "${MODE}" = "gpu" ]; then
    GPU_EXTRA=(
      --cache-dataset-on-device "${CACHE_ON_DEVICE}"
      --refine-ratio "${REFINE_RATIO}"
    )
  fi
  PYTHONUNBUFFERED=1 vectordbbench "${CMD}" \
    --uri "${URI}" \
    --case-type PerformanceCustomDataset \
    --custom-case-name "${CASE_NAME}" \
    --custom-dataset-name "${DATASET_NAME}" \
    --custom-dataset-dir "${DATASET_DIR}" \
    --custom-dataset-size 1000000 \
    --custom-dataset-dim 128 \
    --custom-dataset-metric-type L2 \
    --custom-dataset-file-count 1 \
    --lists "${LISTS}" \
    --probes "${NPROBE}" \
    --k "${K}" \
    --search-serial \
    --skip-search-concurrent \
    --db-label "${DB_LABEL}" \
    "${GPU_EXTRA[@]}" \
    "${EXTRA[@]}" 2>&1 | tee -a "${LOG}"
done

echo ""
echo "VDBBENCH SWEEP OK (${MODE})"
echo "  log: ${LOG}"
echo "Parse QPS/recall (serial search often leaves summary qps=0):"
echo "  grep -E 'nprobe=|search entire test_data' ${LOG}"
echo "  Effective QPS = queries / cost from each 'search entire test_data' line."
if [ "${MODE}" = "gpu" ]; then
  echo "Sealed HIP path check (milvus log):"
  echo "  grep -aE 'GPU_CUVS_IVF_FLAT|InvalidDeviceFunction|IVF_FLAT_CC' \\\\"
  echo "    \$WORKDIR/logs/milvus_gpu_standalone.log | tail -40"
fi
