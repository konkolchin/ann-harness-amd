#!/usr/bin/env bash
# Source this (or call from wrappers) so pip-installed cuVS finds libcuvs_c.so.
# Usage: source scripts/cuvs_pip_ld_path.sh
#
# No conda — paths come from the active venv's site-packages.

_cuvs_pip_ld_path() {
  local site py
  py="$(command -v python3 || true)"
  if [ -z "${py}" ]; then
    echo "ERROR: python3 not on PATH (activate ~/cuvs-bench-venv first)" >&2
    return 1
  fi
  site="$(python3 - <<'PY'
import site
from pathlib import Path
pkgs = []
try:
    pkgs.extend(site.getsitepackages())
except Exception:
    pass
try:
    pkgs.append(site.getusersitepackages())
except Exception:
    pass
# Prefer real directories under the venv
out = []
for p in pkgs:
    if p and Path(p).is_dir():
        out.append(p)
print(out[0] if out else "")
PY
)"
  if [ -z "${site}" ] || [ ! -d "${site}" ]; then
    echo "ERROR: could not resolve site-packages" >&2
    return 1
  fi

  local extras
  extras="$(
    {
      find "${site}" -maxdepth 3 -type f \( -name 'libcuvs*.so*' -o -name 'libraft*.so*' -o -name 'librmm*.so*' \) 2>/dev/null \
        | xargs -r -n1 dirname
      find "${site}/nvidia" -type d \( -name lib -o -name lib64 \) 2>/dev/null
    } | sort -u | paste -sd: -
  )"

  if [ -z "${extras}" ]; then
    echo "WARNING: no libcuvs*/nvidia lib dirs under ${site}" >&2
    echo "  Try: find ~/cuvs-bench-venv -name 'libcuvs_c.so*'" >&2
    return 1
  fi

  export LD_LIBRARY_PATH="${extras}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "LD_LIBRARY_PATH updated for pip cuVS (site=${site})"
}

_cuvs_pip_ld_path
