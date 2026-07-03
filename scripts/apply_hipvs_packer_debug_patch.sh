#!/usr/bin/env bash
# Apply IVF packer debug instrumentation to a local hipVS clone (Layer 1.5).
# Prints list_size, dim, veclen, kIndexGroupSize, mask_true vs expected on first mismatch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="${REPO_ROOT}/patches/hipvs/0001-ivf-packer-debug-instrumentation.patch"
HIPVS_DIR="${1:-${HOME}/rocmds_check_gfx1100/hipVS}"

if [ ! -d "${HIPVS_DIR}/.git" ]; then
  echo "hipVS git checkout not found: ${HIPVS_DIR}" >&2
  exit 1
fi
if [ ! -f "${PATCH}" ]; then
  echo "Patch missing: ${PATCH}" >&2
  exit 1
fi

cd "${HIPVS_DIR}"
if git apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "Patch already applied in ${HIPVS_DIR}"
  exit 0
fi
git apply "${PATCH}"
echo "Applied ${PATCH}"
echo "Rebuild gtests, then run:"
echo "  export LD_LIBRARY_PATH=\${INSTALL_PREFIX}/lib:\${ROCM_PATH}/lib:\$LD_LIBRARY_PATH"
echo "  \$HIPVS_DIR/cpp/build/gtests/NEIGHBORS_ANN_IVF_FLAT_TEST \\"
echo "    --gtest_filter='AnnIVFFlatTest/AnnIVFFlatTestF_float.*' 2>&1 | tee packer_debug.log"
