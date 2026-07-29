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
        'if ( MILVUS_GPU_VERSION )\n'
        "  # GPU Knowhere uses WITH_CUVS. WITH_HIP is AMD-only and MUST stay OFF for CUDA CI:\n"
        "  # never FORCE ON just because MILVUS_GPU_VERSION=ON (that pulls ROCm and breaks NVIDIA).\n"
        "  # Opt-in: -DMILVUS_WITH_HIP=ON  or  export MILVUS_WITH_HIP=1  or  KNOWHERE_WITH_HIP=1\n"
        '  set(WITH_CUVS ON CACHE BOOL "" FORCE )\n'
        "  set(_milvus_want_hip OFF)\n"
        "  if(DEFINED MILVUS_WITH_HIP AND MILVUS_WITH_HIP)\n"
        "    set(_milvus_want_hip ON)\n"
        "  elseif(DEFINED ENV{MILVUS_WITH_HIP})\n"
        '    string(TOUPPER "$ENV{MILVUS_WITH_HIP}" _milvus_hip_env)\n'
        '    if(_milvus_hip_env STREQUAL "1"\n'
        '       OR _milvus_hip_env STREQUAL "ON"\n'
        '       OR _milvus_hip_env STREQUAL "TRUE"\n'
        '       OR _milvus_hip_env STREQUAL "YES")\n'
        "      set(_milvus_want_hip ON)\n"
        "    endif()\n"
        "    unset(_milvus_hip_env)\n"
        "  elseif(DEFINED ENV{KNOWHERE_WITH_HIP})\n"
        '    string(TOUPPER "$ENV{KNOWHERE_WITH_HIP}" _milvus_hip_env)\n'
        '    if(_milvus_hip_env STREQUAL "1"\n'
        '       OR _milvus_hip_env STREQUAL "ON"\n'
        '       OR _milvus_hip_env STREQUAL "TRUE"\n'
        '       OR _milvus_hip_env STREQUAL "YES")\n'
        "      set(_milvus_want_hip ON)\n"
        "    endif()\n"
        "    unset(_milvus_hip_env)\n"
        "  endif()\n"
        "  set(WITH_HIP ${_milvus_want_hip} CACHE BOOL\n"
        '      "Link Knowhere against hipVS/hipRAFT (ROCm); OFF for CUDA" FORCE)\n'
        "  if(_milvus_want_hip)\n"
        '    set(WITH_RAFT OFF CACHE BOOL "" FORCE )\n'
        '    message(STATUS "MILVUS Layer3 HIP: WITH_CUVS=ON WITH_HIP=ON (hipVS/hipRAFT)")\n'
        "  else()\n"
        '    message(STATUS "MILVUS GPU: WITH_CUVS=ON WITH_HIP=OFF (CUDA; set MILVUS_WITH_HIP=1 for AMD)")\n'
        "  endif()\n"
        "  unset(_milvus_want_hip)\n"
        "endif ()\n"
        # If HIP was enabled via -DWITH_HIP=ON / env without entering the GPU block
        # (or GPU flag spelling differs), still force cuVS — Knowhere fatals otherwise.
        "if(WITH_HIP)\n"
        '  set(WITH_CUVS ON CACHE BOOL "" FORCE )\n'
        '  message(STATUS "MILVUS Layer3: WITH_HIP=ON -> forcing WITH_CUVS=ON")\n'
        "endif()\n"
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

    replacement = """if(WITH_HIP)
# Prefer hipVS/hipRAFT + ROCm ahead of Conan boost for find_package(cuvs/raft/hip).
# Do NOT read ENV{INSTALL_PREFIX}: Milvus core_build.sh overwrites it to the Milvus
# output tree, which hides hipVS/xxHash under ~/rocmds_check_gfx1100/install.
# Do NOT prepend Knowhere Conan CMakeDeps generators - their xxHashConfig.cmake
# calls check_build_type_defined and breaks configure outside conan toolchain.
set(_milvus_hip_prefix "$ENV{MILVUS_HIP_INSTALL_PREFIX}")
if(_milvus_hip_prefix STREQUAL "")
  set(_milvus_hip_prefix "$ENV{ROCMDS_INSTALL_PREFIX}")
endif()
if(_milvus_hip_prefix STREQUAL "")
  set(_milvus_hip_prefix "$ENV{HOME}/rocmds_check_gfx1100/install")
endif()
set(_milvus_rocm "$ENV{ROCM_PATH}")
if(_milvus_rocm STREQUAL "")
  set(_milvus_rocm "/opt/rocm")
endif()
set(CMAKE_PREFIX_PATH "${_milvus_hip_prefix};${_milvus_rocm};${CONAN_BOOST_ROOT};${CMAKE_PREFIX_PATH}")
# Standalone xxHash only (installed under hip prefix), never Conan generators.
foreach(_xx_cand
    "${_milvus_hip_prefix}/lib/cmake/xxHash"
    "${_milvus_hip_prefix}/lib64/cmake/xxHash")
  if(EXISTS "${_xx_cand}/xxHashConfig.cmake" OR EXISTS "${_xx_cand}/xxhash-config.cmake")
    set(xxHash_DIR "${_xx_cand}" CACHE PATH "xxHash CMake package dir" FORCE)
    break()
  endif()
endforeach()
message(STATUS "MILVUS Layer3 Knowhere CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "MILVUS Layer3 Knowhere xxHash_DIR: ${xxHash_DIR}")
else()
set( CMAKE_PREFIX_PATH ${CONAN_BOOST_ROOT} )
endif()

# Resolve local Knowhere: -D, env, or default AMD workdir (skip FetchContent / github.dxc.com).
if(NOT MILVUS_KNOWHERE_SOURCE_DIR)
  if(DEFINED ENV{MILVUS_KNOWHERE_SOURCE_DIR} AND NOT "$ENV{MILVUS_KNOWHERE_SOURCE_DIR}" STREQUAL "")
    set(MILVUS_KNOWHERE_SOURCE_DIR "$ENV{MILVUS_KNOWHERE_SOURCE_DIR}")
  elseif(DEFINED ENV{KNOWHERE_DIR} AND NOT "$ENV{KNOWHERE_DIR}" STREQUAL "")
    set(MILVUS_KNOWHERE_SOURCE_DIR "$ENV{KNOWHERE_DIR}")
  elseif(EXISTS "$ENV{HOME}/rocmds_check_gfx1100/knowhere/CMakeLists.txt")
    set(MILVUS_KNOWHERE_SOURCE_DIR "$ENV{HOME}/rocmds_check_gfx1100/knowhere")
  endif()
endif()
if(TARGET knowhere OR TARGET knowhere::knowhere)
  message(STATUS "MILVUS Layer3: knowhere already added; skip second add_subdirectory")
  if(NOT knowhere_SOURCE_DIR AND MILVUS_KNOWHERE_SOURCE_DIR)
    set(knowhere_SOURCE_DIR "${MILVUS_KNOWHERE_SOURCE_DIR}")
  endif()
elseif(MILVUS_KNOWHERE_SOURCE_DIR AND EXISTS "${MILVUS_KNOWHERE_SOURCE_DIR}/CMakeLists.txt")
  message(STATUS "MILVUS Layer3: using local Knowhere SOURCE_DIR=${MILVUS_KNOWHERE_SOURCE_DIR}")
  set(knowhere_SOURCE_DIR "${MILVUS_KNOWHERE_SOURCE_DIR}")
  set(knowhere_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/knowhere-build")
  file(MAKE_DIRECTORY "${knowhere_BINARY_DIR}")
  # Re-assert cuVS before nested configure (Milvus may re-enter this file after project()).
  if(WITH_HIP)
    set(WITH_CUVS ON CACHE BOOL "" FORCE)
  endif()
  add_subdirectory( ${knowhere_SOURCE_DIR} ${knowhere_BINARY_DIR} )
elseif(WITH_HIP)
  message(FATAL_ERROR
    "HIP build requires a local Knowhere tree (FetchContent to github.dxc.com is not used).\n"
    "  cmake ... -DMILVUS_KNOWHERE_SOURCE_DIR=/path/to/knowhere\n"
    "  or: export MILVUS_KNOWHERE_SOURCE_DIR=/path/to/knowhere\n"
    "  got MILVUS_KNOWHERE_SOURCE_DIR='${MILVUS_KNOWHERE_SOURCE_DIR}'")
else()
  message(STATUS "MILVUS Layer3: FetchContent knowhere from ${GIT_REPOSITORY} (${KNOWHERE_VERSION})")
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
        # Always LF: git apply on Linux rejects CRLF as "corrupt patch".
        OUT.write_bytes(diff.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
        if not OUT.read_bytes().endswith(b"\n"):
            OUT.write_bytes(OUT.read_bytes() + b"\n")
        print(f"wrote {OUT} ({len(diff.splitlines())} lines)")
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
