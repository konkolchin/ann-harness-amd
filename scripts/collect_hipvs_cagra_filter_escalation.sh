#!/usr/bin/env bash
# Collect environment + point at escalation doc / GitHub issue template.
#
# Usage on amd-rx7900xtx:
#   source ~/hipvs-bench-venv/bin/activate
#   export WORKDIR=~/rocmds_check_gfx1100
#   bash scripts/collect_hipvs_cagra_filter_escalation.sh
#
# Then open: https://github.com/ROCm-DS/hipVS/issues/new
# Paste body from docs/escalation_hipvs_cagra_filter_gfx1100.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${LOG_DIR}/hipvs_cagra_filter_escalation_env_${TS}.txt"
FILTER_JSON="${FILTER_JSON:-}"

mkdir -p "${LOG_DIR}"

{
  echo "# hipVS CAGRA filter escalation env"
  echo "date=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo ""
  echo "== rocm-smi =="
  rocm-smi --showproductname 2>&1 | head -30 || true
  echo ""
  echo "== rocminfo (gfx) =="
  rocminfo 2>&1 | grep -E 'Name:|Marketing|gfx' | head -30 || true
  echo ""
  echo "== ROCm version =="
  cat /opt/rocm/.info/version 2>&1 || true
  ls /opt/rocm* 2>&1 | head -20 || true
  echo ""
  echo "== Python packages =="
  python3 - <<'PY' 2>&1 || true
import sys
print("python", sys.version)
try:
    import cuvs
    print("cuvs", getattr(cuvs, "__version__", "?"))
except Exception as e:
    print("cuvs import FAIL", e)
try:
    import cupy
    print("cupy", cupy.__version__)
except Exception as e:
    print("cupy import FAIL", e)
try:
    from cuvs.neighbors import cagra, filters
    print("filters", [a for a in dir(filters) if not a.startswith("_")])
except Exception as e:
    print("filters import FAIL", e)
PY
  echo ""
  echo "== latest filter repro JSON =="
  if [ -n "${FILTER_JSON}" ] && [ -f "${FILTER_JSON}" ]; then
    ls -la "${FILTER_JSON}"
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print([(c['tag'], c.get('recall'), c.get('neg1')) for c in d.get('cases',[])])" "${FILTER_JSON}"
  else
    ls -lt "${LOG_DIR}"/lib_hipvs_cagra_filter_*.json 2>/dev/null | head -3 || echo "(none)"
    LATEST=$(ls -t "${LOG_DIR}"/lib_hipvs_cagra_filter_*.json 2>/dev/null | head -1 || true)
    if [ -n "${LATEST}" ]; then
      echo "LATEST=${LATEST}"
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print([(c['tag'], c.get('recall'), c.get('neg1')) for c in d.get('cases',[])])" "${LATEST}"
    fi
  fi
} | tee "${OUT}"

echo ""
echo "Wrote: ${OUT}"
echo ""
echo "File issue:"
echo "  https://github.com/ROCm-DS/hipVS/issues/new"
echo "Template:"
echo "  ${REPO_ROOT}/docs/escalation_hipvs_cagra_filter_gfx1100.md"
echo "Attach: ${OUT} and lib_hipvs_cagra_filter_*.json"
