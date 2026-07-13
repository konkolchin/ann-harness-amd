#!/usr/bin/env python3
"""Generate patches/milvus/0001-knowhere-hip-dxc-fetchcontent.patch against Milvus v2.5.4.

Reads the real upstream file (or MILVUS_CMAKELISTS path) so whitespace matches the tag blob.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "patches" / "milvus" / "0001-knowhere-hip-dxc-fetchcontent.patch"
REL = Path("internal/core/thirdparty/knowhere/CMakeLists.txt")

DEFAULT_SRC_CANDIDATES = [
    Path(os.environ["MILVUS_CMAKELISTS"]) if os.environ.get("MILVUS_CMAKELISTS") else None,
    ROOT.parent / "_tmp_milvus_v254" / REL,
    Path.home() / "rocmds_check_gfx1100" / "milvus" / REL,
]


def load_old() -> str:
    for p in DEFAULT_SRC_CANDIDATES:
        if p is not None and p.is_file():
            text = p.read_text(encoding="utf-8")
            # Normalize to LF for the patch; git apply on Linux expects LF.
            return text.replace("\r\n", "\n").replace("\r", "\n")
    raise SystemExit(
        "Need upstream CMakeLists.txt. Set MILVUS_CMAKELISTS=... or clone milvus v2.5.4 "
        f"so one of these exists: {[str(p) for p in DEFAULT_SRC_CANDIDATES if p]}"
    )


def make_new(old: str) -> str:
    if "MILVUS Layer3 HIP" in old:
        raise SystemExit("source already patched; reset milvus tree first")

    # Match exact upstream lines (4-space indent, double-space after GIT_REPOSITORY).
    old_repo = 'set( KNOWHERE_VERSION v2.5.2 )\nset( GIT_REPOSITORY  "https://github.com/zilliztech/knowhere.git")\n'
    new_repo = """# Layer 3 AMD/HIP: default to DXC Knowhere 2.5 (HIP merged). Override with:
#   -DMILVUS_KNOWHERE_GIT_REPOSITORY=... -DMILVUS_KNOWHERE_VERSION=...
#   -DMILVUS_KNOWHERE_SOURCE_DIR=/path/to/local/knowhere  (skip git fetch)
if(NOT DEFINED MILVUS_KNOWHERE_VERSION)
  set(MILVUS_KNOWHERE_VERSION "2.5")
endif()
if(NOT DEFINED MILVUS_KNOWHERE_GIT_REPOSITORY)
  set(MILVUS_KNOWHERE_GIT_REPOSITORY "https://github.dxc.com/llmkb-internal/knowhere.git")
endif()
set( KNOWHERE_VERSION ${MILVUS_KNOWHERE_VERSION} )
set( GIT_REPOSITORY ${MILVUS_KNOWHERE_GIT_REPOSITORY} )
"""
    if old_repo not in old:
        # Tolerate single-space variant
        old_repo_alt = 'set( KNOWHERE_VERSION v2.5.2 )\nset( GIT_REPOSITORY "https://github.com/zilliztech/knowhere.git")\n'
        if old_repo_alt not in old:
            raise SystemExit("unexpected KNOWHERE_VERSION / GIT_REPOSITORY block")
        old_repo = old_repo_alt

    new = old.replace(old_repo, new_repo, 1)
    # Insert local-source status after Knowhere version message
    marker = 'message(STATUS "Knowhere version: ${KNOWHERE_VERSION}")\n'
    inject = marker + (
        'if(DEFINED MILVUS_KNOWHERE_SOURCE_DIR AND EXISTS "${MILVUS_KNOWHERE_SOURCE_DIR}/CMakeLists.txt")\n'
        '  message(STATUS "Knowhere local source: ${MILVUS_KNOWHERE_SOURCE_DIR}")\n'
        "endif()\n"
    )
    if marker not in new:
        raise SystemExit("missing Knowhere version message")
    new = new.replace(marker, inject, 1)

    old_raft = (
        'if ( MILVUS_GPU_VERSION STREQUAL "ON" )\n'
        "    set(WITH_RAFT ON CACHE BOOL \"\" FORCE )\n"
        "endif ()\n"
    )
    new_raft = (
        'if ( MILVUS_GPU_VERSION STREQUAL "ON" )\n'
        "  # NVIDIA path used WITH_RAFT; AMD Layer-3 HIP Knowhere uses WITH_CUVS + WITH_HIP\n"
        "  # against prebuilt hipVS/hipRAFT under INSTALL_PREFIX.\n"
        '  set(WITH_CUVS ON CACHE BOOL "" FORCE )\n'
        '  set(WITH_HIP ON CACHE BOOL "" FORCE )\n'
        '  set(WITH_RAFT OFF CACHE BOOL "" FORCE )\n'
        '  message(STATUS "MILVUS Layer3 HIP: WITH_CUVS=ON WITH_HIP=ON (hipVS/hipRAFT)")\n'
        "endif ()\n"
    )
    if old_raft not in new:
        old_raft_alt = (
            'if ( MILVUS_GPU_VERSION STREQUAL "ON" )\n'
            ' set(WITH_RAFT ON CACHE BOOL "" FORCE )\n'
            "endif ()\n"
        )
        if old_raft_alt not in new:
            raise SystemExit("unexpected WITH_RAFT block")
        old_raft = old_raft_alt
    new = new.replace(old_raft, new_raft, 1)

    # Replace CMAKE_PREFIX_PATH + FetchContent block with HIP-aware path + optional local source.
    # Capture from set(CMAKE_PREFIX_PATH...) through the FetchContent endif().
    start = new.find("set( CMAKE_PREFIX_PATH ${CONAN_BOOST_ROOT} )")
    if start < 0:
        raise SystemExit("missing CMAKE_PREFIX_PATH ${CONAN_BOOST_ROOT}")
    end_marker = "# get prometheus COMPILE_OPTIONS"
    end = new.find(end_marker, start)
    if end < 0:
        raise SystemExit("missing prometheus COMPILE_OPTIONS marker")

    replacement = """# Prefer hipVS/hipRAFT + ROCm ahead of Conan boost for find_package(cuvs/raft/hip).
set(_milvus_hip_prefix "$ENV{INSTALL_PREFIX}")
if(_milvus_hip_prefix STREQUAL "")
  set(_milvus_hip_prefix "$ENV{HOME}/rocmds_check_gfx1100/install")
endif()
set(_milvus_rocm "$ENV{ROCM_PATH}")
if(_milvus_rocm STREQUAL "")
  set(_milvus_rocm "/opt/rocm")
endif()
set(CMAKE_PREFIX_PATH "${_milvus_hip_prefix};${_milvus_rocm};${CONAN_BOOST_ROOT};${CMAKE_PREFIX_PATH}")
message(STATUS "MILVUS Layer3 Knowhere CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

if(DEFINED MILVUS_KNOWHERE_SOURCE_DIR AND EXISTS "${MILVUS_KNOWHERE_SOURCE_DIR}/CMakeLists.txt")
  set(knowhere_SOURCE_DIR "${MILVUS_KNOWHERE_SOURCE_DIR}")
  set(knowhere_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/knowhere-build")
  file(MAKE_DIRECTORY "${knowhere_BINARY_DIR}")
  add_subdirectory( ${knowhere_SOURCE_DIR} ${knowhere_BINARY_DIR} )
else()
  FetchContent_Declare(
        knowhere
        GIT_REPOSITORY  ${GIT_REPOSITORY}
        GIT_TAG         ${KNOWHERE_VERSION}
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/knowhere-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/knowhere-build
        DOWNLOAD_DIR    ${THIRDPARTY_DOWNLOAD_PATH} )

  FetchContent_GetProperties( knowhere )
  if ( NOT knowhere_POPULATED )
    FetchContent_Populate( knowhere )

    # Adding the following target:
    # knowhere
    add_subdirectory( ${knowhere_SOURCE_DIR}
                      ${knowhere_BINARY_DIR} )
  endif()
endif()

"""
    new = new[:start] + replacement + new[end:]
    return new


def main() -> None:
    old = load_old()
    new = make_new(old)
    td = Path(tempfile.mkdtemp(prefix="milvus_kh_"))
    try:
        (td / REL.parent).mkdir(parents=True)
        (td / REL).write_text(old, encoding="utf-8", newline="\n")
        subprocess.run(["git", "init", "-q"], cwd=td, check=True)
        subprocess.run(["git", "add", str(REL)], cwd=td, check=True)
        subprocess.run(
            ["git", "-c", "user.email=a@b.c", "-c", "user.name=gen", "commit", "-qm", "base"],
            cwd=td,
            check=True,
        )
        (td / REL).write_text(new, encoding="utf-8", newline="\n")
        diff = subprocess.check_output(["git", "diff", "--", str(REL)], cwd=td, text=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(diff, encoding="utf-8", newline="\n")
        print(f"wrote {OUT} ({len(diff.splitlines())} lines)")
    finally:
        shutil.rmtree(td)


if __name__ == "__main__":
    main()
