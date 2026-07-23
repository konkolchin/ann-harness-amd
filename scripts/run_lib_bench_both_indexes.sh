#!/usr/bin/env bash
# One-shot lab capture for the current host (detect AMD vs NVIDIA).
# Run on the physical GPU box after cuvs+cupy import works.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  export WORKDIR="${WORKDIR:-${HOME}/milvus_cuda_4080}"
  bash scripts/run_cuvs_ivf_bench.sh
  INDEX_TYPE=IVF_PQ M=32 bash scripts/run_cuvs_ivf_bench.sh
elif command -v rocm-smi >/dev/null 2>&1; then
  export WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
  bash scripts/run_hipvs_ivf_bench.sh
  INDEX_TYPE=IVF_PQ M=32 bash scripts/run_hipvs_ivf_bench.sh
else
  echo "ERROR: neither nvidia-smi nor rocm-smi found" >&2
  exit 1
fi

echo "Copy newest lib_*.json from ${WORKDIR}/logs into results/lib_bench/ if desired."
