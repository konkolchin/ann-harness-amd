#!/usr/bin/env python3
"""Generate patches/milvus/0005-core-build-cmake-cmd-backslash.patch."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "patches" / "milvus" / "0005-core-build-cmake-cmd-backslash.patch"
MILVUS = ROOT.parent / "_tmp_milvus_v254"


def main() -> None:
    raw = subprocess.check_output(
        ["git", "show", "HEAD:scripts/core_build.sh"], cwd=MILVUS
    )
    text = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode()
    needle = "-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}\n-DCMAKE_BUILD_TYPE"
    if needle not in text:
        raise SystemExit(f"bug pattern not found in core_build.sh: {needle!r}")
    fixed = text.replace(
        needle,
        "-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \\\n-DCMAKE_BUILD_TYPE",
        1,
    )
    td = Path(tempfile.mkdtemp(prefix="milvus_0005_"))
    try:
        (td / "scripts").mkdir()
        (td / "scripts" / "core_build.sh").write_text(text, encoding="utf-8", newline="\n")
        subprocess.run(["git", "init", "-q"], cwd=td, check=True)
        subprocess.run(["git", "add", "scripts/core_build.sh"], cwd=td, check=True)
        subprocess.run(
            ["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "b"],
            cwd=td,
            check=True,
        )
        (td / "scripts" / "core_build.sh").write_text(fixed, encoding="utf-8", newline="\n")
        diff = subprocess.check_output(["git", "diff", "--", "scripts/core_build.sh"], cwd=td)
        OUT.write_bytes(diff if diff.endswith(b"\n") else diff + b"\n")
        print(f"wrote {OUT} ({len(diff.splitlines())} lines)")
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
