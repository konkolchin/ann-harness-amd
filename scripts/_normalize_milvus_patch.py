#!/usr/bin/env python3
"""Normalize a git patch: LF endings + ensure ---/+++ path headers before @@."""
from __future__ import annotations

import sys
from pathlib import Path


def normalize(src: Path, dst: Path) -> None:
    text = src.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode(
        "utf-8", errors="replace"
    )
    lines = text.splitlines(True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            parts = line.strip().split()
            a = (
                parts[2][2:]
                if len(parts) > 2 and parts[2].startswith("a/")
                else (parts[2] if len(parts) > 2 else "")
            )
            b = (
                parts[3][2:]
                if len(parts) > 3 and parts[3].startswith("b/")
                else (parts[3] if len(parts) > 3 else a)
            )
            out.append(line if line.endswith("\n") else line + "\n")
            i += 1
            saw_paths = False
            while i < len(lines) and not lines[i].startswith("diff --git "):
                cur = lines[i]
                if cur.startswith("--- ") or cur.startswith("+++ "):
                    saw_paths = True
                    out.append(cur if cur.endswith("\n") else cur + "\n")
                    i += 1
                    continue
                if cur.startswith("@@") and not saw_paths and a:
                    out.append(f"--- a/{a}\n")
                    out.append(f"+++ b/{b}\n")
                    saw_paths = True
                    out.append(cur if cur.endswith("\n") else cur + "\n")
                    i += 1
                    continue
                out.append(cur if cur.endswith("\n") else cur + "\n")
                i += 1
            continue
        out.append(line if line.endswith("\n") else line + "\n")
        i += 1
    dst.write_text("".join(out), encoding="utf-8", newline="\n")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} SRC DST")
    normalize(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
