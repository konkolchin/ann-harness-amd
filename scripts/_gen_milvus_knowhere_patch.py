#!/usr/bin/env python3
"""Generate patches/milvus/0001-knowhere-hip-dxc-fetchcontent.patch against Milvus v2.5.4."""
from pathlib import Path
import subprocess
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "patches" / "milvus" / "0001-knowhere-hip-dxc-fetchcontent.patch"

OLD = r'''#-------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.
#-------------------------------------------------------------------------------

# Update KNOWHERE_VERSION for the first occurrence
milvus_add_pkg_config("knowhere")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "")
set( KNOWHERE_VERSION v2.5.2 )
set( GIT_REPOSITORY "https://github.com/zilliztech/knowhere.git")
message(STATUS "Knowhere repo: ${GIT_REPOSITORY}")
message(STATUS "Knowhere version: ${KNOWHERE_VERSION}")

message(STATUS "Building knowhere-${KNOWHERE_SOURCE_VER} from source")
message(STATUS ${CMAKE_BUILD_TYPE})

if ( BUILD_DISK_ANN STREQUAL "ON" )
 set(WITH_DISKANN ON CACHE BOOL "" FORCE )
else ()
 set(WITH_DISKANN OFF CACHE BOOL "" FORCE )
endif ()

if ( INDEX_ENGINE STREQUAL "cardinal" )
 set(WITH_CARDINAL ON CACHE BOOL "" FORCE )
else ()
 set(WITH_CARDINAL OFF CACHE BOOL "" FORCE )
endif()

if ( MILVUS_GPU_VERSION STREQUAL "ON" )
 set(WITH_RAFT ON CACHE BOOL "" FORCE )
endif ()

set( CMAKE_PREFIX_PATH ${CONAN_BOOST_ROOT} )
FetchContent_Declare(
 knowhere
 GIT_REPOSITORY ${GIT_REPOSITORY}
 GIT_TAG ${KNOWHERE_VERSION}
 SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/knowhere-src
 BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/knowhere-build
 DOWNLOAD_DIR ${THIRDPARTY_DOWNLOAD_PATH} )

FetchContent_GetProperties( knowhere )
if ( NOT knowhere_POPULATED )
 FetchContent_Populate( knowhere )

 # Adding the following target:
 # knowhere
 add_subdirectory( ${knowhere_SOURCE_DIR}
 ${knowhere_BINARY_DIR} )
endif()

# get prometheus COMPILE_OPTIONS
get_property( var DIRECTORY "${knowhere_SOURCE_DIR}" PROPERTY COMPILE_OPTIONS )
message( STATUS "knowhere src compile options: ${var}" )

set( KNOWHERE_INCLUDE_DIR ${knowhere_SOURCE_DIR}/include CACHE INTERNAL "Path to knowhere include directory" )
'''

NEW = r'''#-------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.
#-------------------------------------------------------------------------------

# Update KNOWHERE_VERSION for the first occurrence
milvus_add_pkg_config("knowhere")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "")
# Layer 3 AMD/HIP: default to DXC Knowhere 2.5 (HIP merged). Override with:
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
message(STATUS "Knowhere repo: ${GIT_REPOSITORY}")
message(STATUS "Knowhere version: ${KNOWHERE_VERSION}")
if(DEFINED MILVUS_KNOWHERE_SOURCE_DIR AND EXISTS "${MILVUS_KNOWHERE_SOURCE_DIR}/CMakeLists.txt")
  message(STATUS "Knowhere local source: ${MILVUS_KNOWHERE_SOURCE_DIR}")
endif()

message(STATUS "Building knowhere-${KNOWHERE_SOURCE_VER} from source")
message(STATUS ${CMAKE_BUILD_TYPE})

if ( BUILD_DISK_ANN STREQUAL "ON" )
 set(WITH_DISKANN ON CACHE BOOL "" FORCE )
else ()
 set(WITH_DISKANN OFF CACHE BOOL "" FORCE )
endif ()

if ( INDEX_ENGINE STREQUAL "cardinal" )
 set(WITH_CARDINAL ON CACHE BOOL "" FORCE )
else ()
 set(WITH_CARDINAL OFF CACHE BOOL "" FORCE )
endif()

if ( MILVUS_GPU_VERSION STREQUAL "ON" )
  # NVIDIA path used WITH_RAFT; AMD Layer-3 HIP Knowhere uses WITH_CUVS + WITH_HIP
  # against prebuilt hipVS/hipRAFT under INSTALL_PREFIX.
  set(WITH_CUVS ON CACHE BOOL "" FORCE )
  set(WITH_HIP ON CACHE BOOL "" FORCE )
  set(WITH_RAFT OFF CACHE BOOL "" FORCE )
  message(STATUS "MILVUS Layer3 HIP: WITH_CUVS=ON WITH_HIP=ON (hipVS/hipRAFT)")
endif ()

# Prefer hipVS/hipRAFT + ROCm ahead of Conan boost for find_package(cuvs/raft/hip).
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
   GIT_REPOSITORY ${GIT_REPOSITORY}
   GIT_TAG ${KNOWHERE_VERSION}
   SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/knowhere-src
   BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/knowhere-build
   DOWNLOAD_DIR ${THIRDPARTY_DOWNLOAD_PATH} )

  FetchContent_GetProperties( knowhere )
  if ( NOT knowhere_POPULATED )
   FetchContent_Populate( knowhere )

   # Adding the following target:
   # knowhere
   add_subdirectory( ${knowhere_SOURCE_DIR}
   ${knowhere_BINARY_DIR} )
  endif()
endif()

# get prometheus COMPILE_OPTIONS
get_property( var DIRECTORY "${knowhere_SOURCE_DIR}" PROPERTY COMPILE_OPTIONS )
message( STATUS "knowhere src compile options: ${var}" )

set( KNOWHERE_INCLUDE_DIR ${knowhere_SOURCE_DIR}/include CACHE INTERNAL "Path to knowhere include directory" )
'''

def main() -> None:
    td = Path(tempfile.mkdtemp(prefix="milvus_kh_"))
    rel = Path("internal/core/thirdparty/knowhere/CMakeLists.txt")
    (td / rel.parent).mkdir(parents=True)
    (td / rel).write_text(OLD, encoding="utf-8", newline="\n")
    subprocess.run(["git", "init", "-q"], cwd=td, check=True)
    subprocess.run(["git", "add", str(rel)], cwd=td, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b.c", "-c", "user.name=gen", "commit", "-qm", "base"],
        cwd=td,
        check=True,
    )
    (td / rel).write_text(NEW, encoding="utf-8", newline="\n")
    diff = subprocess.check_output(["git", "diff", "--", str(rel)], cwd=td, text=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(diff, encoding="utf-8", newline="\n")
    shutil.rmtree(td)
    print(f"wrote {OUT} ({len(diff.splitlines())} lines)")


if __name__ == "__main__":
    main()
