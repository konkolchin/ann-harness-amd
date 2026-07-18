#!/usr/bin/env bash
# Fair CPU-vs-GPU compare: VectorDBBench SIFT-1M IVF nprobe sweep.
#
# MODE=cpu  -> vectordbbench milvusivfflat   (Docker CPU Milvus IVF_FLAT)
# MODE=gpu  -> vectordbbench milvusgpuivfflat (HIP/CUDA GPU Milvus GPU_IVF_FLAT)
#
# Default search stage: CONCURRENT (multi-client QPS) — not serial.
# Serial one-query RPCs under-use GPU; use SEARCH_STAGE=serial only for latency/recall.
#
# Usage:
#   MODE=cpu bash scripts/run_vdbbench_milvus_ivf_sweep.sh
#   MODE=gpu bash scripts/run_vdbbench_milvus_ivf_sweep.sh
#   SEARCH_STAGE=both MODE=gpu ...   # serial recall + concurrent QPS
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
CACHE_ON_DEVICE="${CACHE_ON_DEVICE:-false}"
REFINE_RATIO="${REFINE_RATIO:-1.0}"
# concurrent | serial | both
SEARCH_STAGE="${SEARCH_STAGE:-concurrent}"
NUM_CONCURRENCY="${NUM_CONCURRENCY:-1,10,20,40,80}"
CONCURRENCY_DURATION="${CONCURRENCY_DURATION:-30}"

case "${MODE}" in
  cpu) CMD=milvusivfflat ;;
  gpu) CMD=milvusgpuivfflat ;;
  *)
    echo "ERROR: MODE must be cpu or gpu (got: ${MODE})" >&2
    exit 1
    ;;
esac

case "${SEARCH_STAGE}" in
  concurrent)
    SEARCH_FLAGS=(--skip-search-serial --search-concurrent)
    ;;
  serial)
    SEARCH_FLAGS=(--search-serial --skip-search-concurrent)
    ;;
  both)
    SEARCH_FLAGS=(--search-serial --search-concurrent)
    ;;
  *)
    echo "ERROR: SEARCH_STAGE must be concurrent|serial|both (got: ${SEARCH_STAGE})" >&2
    exit 1
    ;;
esac

LOG="${LOG:-${LOG_DIR}/vdb_${MODE}_${SEARCH_STAGE}_nprobe_${TS}.log}"
mkdir -p "${LOG_DIR}"

milvus_up() {
  curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1 \
    || (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1
}

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
    echo "  See docs/vdbbench_cpu_gpu_compare.md (parquet prep)" >&2
    exit 1
  fi
done

if ! milvus_up; then
  echo "ERROR: Milvus not reachable on ${URI} (:19530 / :9091)" >&2
  if [ "${MODE}" = "cpu" ]; then
    echo "  Start Docker CPU Milvus (milvus-docker / compose) first." >&2
  else
    echo "  Start HIP Milvus with ROCR_VISIBLE_DEVICES=0 first." >&2
  fi
  exit 1
fi

echo "==> MODE=${MODE} CMD=${CMD} SEARCH_STAGE=${SEARCH_STAGE}"
echo "==> URI=${URI} lists=${LISTS} k=${K} nprobes=${NPROBES}"
echo "==> num-concurrency=${NUM_CONCURRENCY} duration=${CONCURRENCY_DURATION}s"
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

  if ! milvus_up; then
    echo "ERROR: Milvus went down before nprobe=${NPROBE}. Check docker/HIP logs." >&2
    exit 1
  fi

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

  STEP_LOG="$(mktemp)"
  set +e
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
    --num-concurrency "${NUM_CONCURRENCY}" \
    --concurrency-duration "${CONCURRENCY_DURATION}" \
    "${SEARCH_FLAGS[@]}" \
    --db-label "${DB_LABEL}" \
    "${GPU_EXTRA[@]}" \
    "${EXTRA[@]}" 2>&1 | tee -a "${LOG}" | tee "${STEP_LOG}"
  pipe_rc=${PIPESTATUS[0]}
  set -e

  if [ "${pipe_rc}" -ne 0 ] \
    || grep -qE 'failed to run|Connection refused|Fail connecting to server|Server unavailable' "${STEP_LOG}" \
    || grep -qE '\| x[[:space:]]*$' "${STEP_LOG}"; then
    echo "ERROR: VectorDBBench case failed for nprobe=${NPROBE} (see ${LOG})" >&2
    if ! milvus_up; then
      echo "  Milvus is down on :19530 after this step." >&2
    fi
    rm -f "${STEP_LOG}"
    exit 1
  fi

  ok_search=0
  if [ "${SEARCH_STAGE}" = "serial" ] || [ "${SEARCH_STAGE}" = "both" ]; then
    grep -q 'search entire test_data' "${STEP_LOG}" && ok_search=1
  fi
  if [ "${SEARCH_STAGE}" = "concurrent" ] || [ "${SEARCH_STAGE}" = "both" ]; then
    # Concurrent stage reports QPS in metrics / conc lists / task summary.
    if grep -qE 'conc_qps|concurrent search|Concurrency|max_qps|search_concurrent' "${STEP_LOG}" \
      || grep -qE '\| [1-9][0-9]*\.[0-9]+ +[0-9]' "${STEP_LOG}"; then
      ok_search=1
    fi
    # Summary line with non-zero qps column (between load_dur and latency)
    if grep -E 'Milvus \|' "${STEP_LOG}" | grep -qvE '\| 0\.0 +0\.0 +'; then
      ok_search=1
    fi
  fi
  if [ "${ok_search}" -ne 1 ]; then
    echo "ERROR: no successful ${SEARCH_STAGE} search evidence for nprobe=${NPROBE}" >&2
    rm -f "${STEP_LOG}"
    exit 1
  fi
  rm -f "${STEP_LOG}"
done

echo ""
echo "VDBBENCH SWEEP OK (${MODE}, ${SEARCH_STAGE})"
echo "  log: ${LOG}"
if [ "${SEARCH_STAGE}" = "serial" ]; then
  echo "Parse serial QPS/recall:"
  echo "  grep -E 'nprobe=|search entire test_data' ${LOG}"
else
  echo "Parse concurrent QPS (peak across concurrency levels):"
  echo "  grep -E 'nprobe=|qps|concurrency|conc_' ${LOG} | head -80"
  echo "  Prefer the task-summary qps column or max conc_qps per nprobe."
fi
if [ "${MODE}" = "gpu" ]; then
  echo "Sealed HIP path check:"
  echo "  grep -aE 'GPU_CUVS_IVF_FLAT|InvalidDeviceFunction' \\\\"
  echo "    \$WORKDIR/logs/milvus_gpu_standalone.log | tail -40"
fi
