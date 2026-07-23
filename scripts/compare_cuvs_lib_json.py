#!/usr/bin/env python3
"""Compare two library-bench JSON files (hipVS vs cuVS). Prints speed-up table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hipvs", required=True, help="hipVS results JSON")
    p.add_argument("--cuvs", required=True, help="cuVS results JSON")
    args = p.parse_args()

    a = load(Path(args.hipvs))
    b = load(Path(args.cuvs))
    if a.get("index_type") != b.get("index_type"):
        raise SystemExit(f"index_type mismatch: {a.get('index_type')} vs {b.get('index_type')}")

    print(
        f"index={a.get('index_type')}  "
        f"hipVS={a.get('gpu_name')}  cuVS={b.get('gpu_name')}"
    )
    print(f"{'nprobe':>6}  {'hipVS QPS':>10}  {'cuVS QPS':>10}  {'speed-up':>10}  "
          f"{'R@10 hip':>8}  {'R@10 cu':>8}")
    by_np_b = {r["nprobe"]: r for r in b["nprobe_results"]}
    for ra in a["nprobe_results"]:
        npv = ra["nprobe"]
        rb = by_np_b[npv]
        su = ra["qps"] / rb["qps"] if rb["qps"] else float("nan")
        rh = ra.get("recall@10", float("nan"))
        rc = rb.get("recall@10", float("nan"))
        print(
            f"{npv:6d}  {ra['qps']:10.1f}  {rb['qps']:10.1f}  {su:9.2f}x  "
            f"{rh:8.4f}  {rc:8.4f}"
        )


if __name__ == "__main__":
    main()
