from pathlib import Path
import subprocess

mil = Path(r"c:\CUDA-to-Hip\_tmp_milvus_v254")
rel = Path("internal/core/src/CMakeLists.txt")
p = mil / rel
subprocess.run(["git", "checkout", "--", str(rel)], cwd=mil, check=True)
t = p.read_text(encoding="utf-8").replace("\r\n", "\n")
old = (
    'if(MILVUS_GPU_VERSION STREQUAL "ON")\n'
    "    project(core CXX C CUDA)\n"
    "else()\n"
    "    project(core CXX C)\n"
    "endif()\n"
)
new = (
    "# Layer 3 AMD/HIP: GPU indexes come from HIP Knowhere/hipVS — do not enable\n"
    "# the CUDA language (requires nvcc). Keep MILVUS_GPU_VERSION for index registration.\n"
    'if(MILVUS_GPU_VERSION STREQUAL "ON")\n'
    '    message(STATUS "MILVUS Layer3 HIP: project(core CXX C) without CUDA language")\n'
    "endif()\n"
    "project(core CXX C)\n"
)
if old not in t:
    raise SystemExit(f"old block not found:\n{t[400:900]!r}")
p.write_text(t.replace(old, new, 1), encoding="utf-8", newline="\n")
diff = subprocess.check_output(["git", "diff", "--", str(rel)], cwd=mil, text=True)
out = Path(r"c:\CUDA-to-Hip\ann_harness_repo\patches\milvus\0002-skip-cuda-language-for-hip.patch")
out.write_text(diff, encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(diff.splitlines())} lines)")
print(diff)
