#!/usr/bin/env python3
"""Generate milvus Layer-3 API/clang compat patches against v2.5.4."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "patches" / "milvus"
BASE = "https://raw.githubusercontent.com/milvus-io/milvus/v2.5.4/"
RELS = [
    "internal/core/src/index/VectorDiskIndex.cpp",
    "internal/core/src/index/VectorMemIndex.cpp",
    "internal/core/src/index/InvertedIndexTantivy.cpp",
]


def fetch(rel: str) -> str:
    data = urllib.request.urlopen(BASE + rel, timeout=60).read()
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8")


def git(td: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=td, check=True)


def write_diff(td: Path, out: Path, paths: list[str]) -> None:
    diff = subprocess.check_output(["git", "diff", "--", *paths], cwd=td, text=True)
    out.write_text(diff, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(diff.splitlines())} lines)")


def main() -> None:
    td = Path(tempfile.mkdtemp(prefix="milvus_api_"))
    try:
        git(td, "init", "-q")
        texts: dict[str, str] = {}
        for rel in RELS:
            text = fetch(rel)
            texts[rel] = text
            p = td / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
        git(td, "add", ".")
        git(td, "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-qm", "base")

        disk = texts[RELS[0]]
        old_disk = (
            "    auto opt_fields = GetValueFromConfig<OptFieldT>(config, VEC_OPT_FIELDS);\n"
            "    if (opt_fields.has_value() && index_.IsAdditionalScalarSupported()) {\n"
            "        build_config[VEC_OPT_FIELDS_PATH] =\n"
            "            file_manager_->CacheOptFieldToDisk(opt_fields.value());\n"
        )
        new_disk = (
            "    auto opt_fields = GetValueFromConfig<OptFieldT>(config, VEC_OPT_FIELDS);\n"
            "    // Knowhere 2.5+: IsAdditionalScalarSupported(bool is_mv_only)\n"
            "    auto is_partition_key_isolation =\n"
            '        GetValueFromConfig<bool>(build_config, "partition_key_isolation");\n'
            "    if (opt_fields.has_value() &&\n"
            "        index_.IsAdditionalScalarSupported(\n"
            "            is_partition_key_isolation.value_or(false))) {\n"
            "        build_config[VEC_OPT_FIELDS_PATH] =\n"
            "            file_manager_->CacheOptFieldToDisk(opt_fields.value());\n"
        )
        if old_disk not in disk:
            raise SystemExit("VectorDiskIndex pattern missing")
        (td / RELS[0]).write_text(disk.replace(old_disk, new_disk, 1), encoding="utf-8", newline="\n")

        mem = texts[RELS[1]]
        old_mem = (
            "    auto opt_fields = GetValueFromConfig<OptFieldT>(config, VEC_OPT_FIELDS);\n"
            "    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info;\n"
            "    if (opt_fields.has_value() && index_.IsAdditionalScalarSupported() &&\n"
            '        config.value("partition_key_isolation", false)) {\n'
            "        scalar_info = file_manager_->CacheOptFieldToMemory(opt_fields.value());\n"
            "    }\n"
        )
        new_mem = (
            "    auto opt_fields = GetValueFromConfig<OptFieldT>(config, VEC_OPT_FIELDS);\n"
            "    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info;\n"
            "    // Knowhere 2.5+: IsAdditionalScalarSupported(bool is_mv_only)\n"
            "    auto is_partition_key_isolation =\n"
            '        GetValueFromConfig<bool>(config, "partition_key_isolation");\n'
            "    if (opt_fields.has_value() &&\n"
            "        index_.IsAdditionalScalarSupported(\n"
            "            is_partition_key_isolation.value_or(false))) {\n"
            "        scalar_info = file_manager_->CacheOptFieldToMemory(opt_fields.value());\n"
            "    }\n"
        )
        if old_mem not in mem:
            raise SystemExit("VectorMemIndex pattern missing")
        (td / RELS[1]).write_text(mem.replace(old_mem, new_mem, 1), encoding="utf-8", newline="\n")

        write_diff(td, OUT_DIR / "0003-knowhere-is-additional-scalar-mv-only.patch", [RELS[0], RELS[1]])

        git(td, "checkout", "--", ".")
        tantivy = texts[RELS[2]]
        t2 = tantivy.replace(
            "wrapper_->template add_multi_data(", "wrapper_->add_multi_data("
        ).replace(
            "wrapper_->template add_multi_data_by_single_segment_writer(",
            "wrapper_->add_multi_data_by_single_segment_writer(",
        )
        if t2 == tantivy:
            raise SystemExit("no tantivy template replacements")
        (td / RELS[2]).write_text(t2, encoding="utf-8", newline="\n")
        write_diff(td, OUT_DIR / "0004-clang-tantivy-drop-spurious-template-kw.patch", [RELS[2]])

        git(td, "checkout", "--", ".")
        for name in (
            "0003-knowhere-is-additional-scalar-mv-only.patch",
            "0004-clang-tantivy-drop-spurious-template-kw.patch",
        ):
            patch = OUT_DIR / name
            r = subprocess.run(
                ["git", "apply", "--check", str(patch)],
                cwd=td,
                capture_output=True,
                text=True,
            )
            print(name, "check", r.returncode, (r.stderr or r.stdout).strip())
            if r.returncode != 0:
                raise SystemExit(f"apply --check failed for {name}")
            git(td, "apply", str(patch))
        print("OK: both patches apply cleanly to v2.5.4")
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
