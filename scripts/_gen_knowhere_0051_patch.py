#!/usr/bin/env python3
"""Generate patches/knowhere/0051-libhipcuvs-milvus-hip-prefix.patch."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "patches" / "knowhere" / "0051-libhipcuvs-milvus-hip-prefix.patch"
KH = ROOT.parent / "_tmp_knowhere_fix"
REL = Path("cmake/libs/libhipcuvs.cmake")


def main() -> None:
    text = (KH / REL).read_text(encoding="utf-8").replace("\r\n", "\n")
    needle = (
        "if(DEFINED ENV{INSTALL_PREFIX})\n"
        '  list(PREPEND CMAKE_PREFIX_PATH "$ENV{INSTALL_PREFIX}")\n'
        "endif()\n"
    )
    insert = (
        "# Milvus Layer-3 unsets INSTALL_PREFIX before configure (own install layout).\n"
        "# Prefer HIPVS_PREFIX / MILVUS_HIP_INSTALL_PREFIX / ROCMDS_INSTALL_PREFIX.\n"
        "if(DEFINED ENV{MILVUS_HIP_INSTALL_PREFIX} AND NOT \"$ENV{MILVUS_HIP_INSTALL_PREFIX}\" STREQUAL \"\")\n"
        '  list(PREPEND CMAKE_PREFIX_PATH "$ENV{MILVUS_HIP_INSTALL_PREFIX}")\n'
        "endif()\n"
        "if(DEFINED ENV{ROCMDS_INSTALL_PREFIX} AND NOT \"$ENV{ROCMDS_INSTALL_PREFIX}\" STREQUAL \"\")\n"
        '  list(PREPEND CMAKE_PREFIX_PATH "$ENV{ROCMDS_INSTALL_PREFIX}")\n'
        "endif()\n"
        + needle
    )
    if needle not in text:
        raise SystemExit("INSTALL_PREFIX block not found in libhipcuvs.cmake")
    if "MILVUS_HIP_INSTALL_PREFIX" in text:
        raise SystemExit("already patched")
    fixed = text.replace(needle, insert, 1)
    td = Path(tempfile.mkdtemp(prefix="kh_0051_"))
    try:
        (td / REL.parent).mkdir(parents=True)
        (td / REL).write_text(text, encoding="utf-8", newline="\n")
        subprocess.run(["git", "init", "-q"], cwd=td, check=True)
        subprocess.run(["git", "add", str(REL)], cwd=td, check=True)
        subprocess.run(
            ["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "b"],
            cwd=td,
            check=True,
        )
        (td / REL).write_text(fixed, encoding="utf-8", newline="\n")
        diff = subprocess.check_output(["git", "diff", "--", str(REL)], cwd=td)
        OUT.write_bytes(diff if diff.endswith(b"\n") else diff + b"\n")
        print(f"wrote {OUT} ({len(diff.splitlines())} lines)")
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
