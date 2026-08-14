#!/usr/bin/env bash
# A/B hipVS IVF-PQ launch variants on gfx1100 (requires patch 0005 applied + rebuilt).
#
# Default grid: 3 variants × {auto,256,512} block threads, plus no-local-topk once.
# Override with SWEEP_CASES (newline-separated "tag|exports").
#
# Usage:
#   source ~/hipvs-bench-venv/bin/activate
#   bash scripts/run_hipvs_ivf_pq_launch_knob_sweep.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/launch_knobs}"
mkdir -p "${LOG_DIR}"

# Unset knobs between cases so leftovers don't stick
unset_knobs() {
  unset HIPVS_IVF_PQ_FORCE_VARIANT \
        HIPVS_IVF_PQ_FORCE_SMEM_LUT \
        HIPVS_IVF_PQ_FORCE_PRECOMP \
        HIPVS_IVF_PQ_BLOCK_THREADS \
        HIPVS_IVF_PQ_FORCE_NO_LOCAL_TOPK \
        HIPVS_IVF_PQ_PREFERRED_CARVEOUT || true
}

if [[ -z "${SWEEP_CASES:-}" ]]; then
  SWEEP_CASES="$(cat <<'EOF'
baseline|
fast|HIPVS_IVF_PQ_FORCE_VARIANT=fast
no_basediff|HIPVS_IVF_PQ_FORCE_VARIANT=no_basediff
no_smem_lut|HIPVS_IVF_PQ_FORCE_VARIANT=no_smem_lut
fast_bt256|HIPVS_IVF_PQ_FORCE_VARIANT=fast HIPVS_IVF_PQ_BLOCK_THREADS=256
fast_bt512|HIPVS_IVF_PQ_FORCE_VARIANT=fast HIPVS_IVF_PQ_BLOCK_THREADS=512
no_smem_bt256|HIPVS_IVF_PQ_FORCE_VARIANT=no_smem_lut HIPVS_IVF_PQ_BLOCK_THREADS=256
no_local_topk|HIPVS_IVF_PQ_FORCE_VARIANT=fast HIPVS_IVF_PQ_FORCE_NO_LOCAL_TOPK=1
EOF
)"
fi

SUMMARY="${LOG_DIR}/SUMMARY_$(date +%Y%m%d_%H%M%S).txt"
echo "Writing summary to ${SUMMARY}"
{
  echo "# hipVS IVF-PQ launch knob sweep"
  echo "# $(date -Is)"
  echo ""
} >"${SUMMARY}"

while IFS= read -r line; do
  [[ -z "${line}" || "${line}" =~ ^# ]] && continue
  tag="${line%%|*}"
  exports="${line#*|}"
  unset_knobs
  if [[ -n "${exports}" ]]; then
    # exports: space-separated KEY=VAL tokens
    # shellcheck disable=SC2086
    export ${exports}
  fi

  echo ""
  echo "======== CASE ${tag} | ${exports:-'(stock auto)'} ========"
  RESULTS_JSON="${LOG_DIR}/lib_hipvs_ivf_pq_m32_${tag}_$(date +%Y%m%d_%H%M%S).json"
  export RESULTS_JSON
  export INDEX_TYPE=IVF_PQ
  export M="${M:-32}"
  export LUT_DTYPE="${LUT_DTYPE:-float32}"
  export P99_SAMPLE="${P99_SAMPLE:-0}"

  set +e
  bash "${REPO_ROOT}/scripts/run_hipvs_ivf_bench.sh"
  rc=$?
  set -e

  echo "${tag} rc=${rc} json=${RESULTS_JSON} exports=${exports:-stock}" | tee -a "${SUMMARY}"
  if [[ ${rc} -ne 0 ]]; then
    echo "WARN: case ${tag} failed (rc=${rc}); continuing" | tee -a "${SUMMARY}"
  fi
done <<<"${SWEEP_CASES}"

unset_knobs
echo ""
echo "Done. Summary: ${SUMMARY}"
echo "Compare QPS with: python3 scripts/compare_cuvs_lib_json.py ${LOG_DIR}/lib_hipvs_ivf_pq_m32_*.json"
