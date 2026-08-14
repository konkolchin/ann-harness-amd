#!/usr/bin/env python3
"""Print QPS/recall tables from one or more lib-bench JSON files (launch-knob sweep)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsons", nargs="+", type=Path, help="lib_*_*.json result files")
    args = p.parse_args()

    for path in args.jsons:
        if not path.is_file():
            print(f"SKIP missing {path}", file=sys.stderr)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        print(path.name)
        for r in d.get("nprobe_results", []):
            qps = r.get("qps", float("nan"))
            rec = r.get("recall@10", float("nan"))
            print(f"  nprobe={r['nprobe']:2d}  qps={qps:10.0f}  R@10={rec:.4f}")
        print()


if __name__ == "__main__":
    main()
