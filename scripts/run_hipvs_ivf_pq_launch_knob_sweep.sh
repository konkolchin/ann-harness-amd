#!/usr/bin/env bash
# A/B hipVS IVF-PQ launch variants on gfx1100 (requires patch 0005 applied + rebuilt).
#
# Default grid: variants × block threads. Do NOT include FORCE_NO_LOCAL_TOPK —
# flipping manage_local_topk only in compute_similarity_select desyncs the
# search path (caller still expects fused local top-k buffers) → hipError 700.
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
  elif [[ -f "${RESULTS_JSON}" ]]; then
    python3 - "${RESULTS_JSON}" "${tag}" >>"${SUMMARY}" <<'PY'
import json, sys
path, tag = sys.argv[1], sys.argv[2]
d = json.load(open(path))
parts = [f"{tag}"]
for r in d.get("nprobe_results", []):
    parts.append(f"nprobe={r['nprobe']}:qps={r['qps']:.0f}:R10={r.get('recall@10', float('nan')):.4f}")
print("  " + "  ".join(parts))
PY
  fi
done <<<"${SWEEP_CASES}"

unset_knobs
echo ""
echo "Done. Summary: ${SUMMARY}"
echo "Paste SUMMARY or:"
echo "  for f in ${LOG_DIR}/lib_hipvs_ivf_pq_m32_*.json; do"
echo "    python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1].split('/')[-1], [(r['nprobe'], round(r['qps']), round(r.get('recall@10',0),4)) for r in d['nprobe_results']])\" \"\$f\""
echo "  done"
