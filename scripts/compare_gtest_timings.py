#!/usr/bin/env python3
"""Compare hipVS vs cuVS shared-gtest JSON from time_cuvs_gtests.py.

Manager one-liner:
  same unit tests both libs ship; report pass rates + median/geo-mean time ratio
  (hipVS_ms / cuVS_ms). Peer GPUs (7900 XTX vs 4080), not same silicon.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def geo_mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hipvs", type=Path, required=True)
    ap.add_argument("--cuvs", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    h = load(args.hipvs)
    c = load(args.cuvs)

    h_map = {x["name"]: x for x in h.get("cases", [])}
    c_map = {x["name"]: x for x in c.get("cases", [])}
    common = sorted(set(h_map) & set(c_map))
    only_h = sorted(set(h_map) - set(c_map))
    only_c = sorted(set(c_map) - set(h_map))

    ratios: list[float] = []
    rows: list[dict] = []
    both_pass = 0
    for name in common:
        hm, cm = h_map[name], c_map[name]
        if hm["status"] == "PASS" and cm["status"] == "PASS":
            both_pass += 1
        if cm["ms"] <= 0:
            continue
        if hm["status"] != "PASS" or cm["status"] != "PASS":
            continue
        r = hm["ms"] / cm["ms"]
        ratios.append(r)
        rows.append(
            {
                "name": name,
                "hipvs_ms": hm["ms"],
                "cuvs_ms": cm["ms"],
                "ratio_hip_over_cu": r,
            }
        )

    rows.sort(key=lambda r: r["ratio_hip_over_cu"], reverse=True)

    summary = {
        "protocol": "shared_gtest_wallclock_compare",
        "hipvs": {
            "path": str(args.hipvs),
            "host": h.get("host"),
            "gpu": h.get("gpu_name"),
            "n_pass": h.get("n_pass"),
            "n_fail": h.get("n_fail"),
            "n_cases": h.get("n_cases"),
            "sum_case_ms": h.get("sum_case_ms"),
            "suite_wall_s": h.get("suite_wall_s"),
        },
        "cuvs": {
            "path": str(args.cuvs),
            "host": c.get("host"),
            "gpu": c.get("gpu_name"),
            "n_pass": c.get("n_pass"),
            "n_fail": c.get("n_fail"),
            "n_cases": c.get("n_cases"),
            "sum_case_ms": c.get("sum_case_ms"),
            "suite_wall_s": c.get("suite_wall_s"),
        },
        "n_common": len(common),
        "n_both_pass": both_pass,
        "n_only_hipvs": len(only_h),
        "n_only_cuvs": len(only_c),
        "n_timed_both_pass": len(ratios),
        "median_ratio_hip_over_cu": statistics.median(ratios) if ratios else None,
        "geomean_ratio_hip_over_cu": geo_mean(ratios) if ratios else None,
        "sum_ms_ratio_hip_over_cu": (
            (h.get("sum_case_ms") or 0) / (c.get("sum_case_ms") or 1)
            if c.get("sum_case_ms")
            else None
        ),
        "slowest_10_hip_vs_cu": rows[:10],
        "fastest_10_hip_vs_cu": list(reversed(rows[-10:])) if rows else [],
    }

    med = summary["median_ratio_hip_over_cu"]
    geo = summary["geomean_ratio_hip_over_cu"]
    print("=== Shared gtest timing (hipVS / cuVS) ===")
    print(f"hipVS: {h.get('n_pass')}/{h.get('n_cases')} pass  sum_ms={h.get('sum_case_ms')}  {h.get('gpu_name')}")
    print(f"cuVS:  {c.get('n_pass')}/{c.get('n_cases')} pass  sum_ms={c.get('sum_case_ms')}  {c.get('gpu_name')}")
    print(f"common cases: {len(common)}  both PASS timed: {len(ratios)}")
    if med is not None:
        print(f"median  time ratio (hip/cu): {med:.3f}x")
        print(f"geomean time ratio (hip/cu): {geo:.3f}x")
        print(f"sum(ms) ratio (hip/cu):      {summary['sum_ms_ratio_hip_over_cu']:.3f}x")
        # Manager sentence
        if geo < 0.9:
            verdict = "hipVS faster on this suite (peer GPUs)."
        elif geo <= 1.25:
            verdict = "hipVS roughly competitive with cuVS on this suite (peer GPUs)."
        elif geo <= 2.0:
            verdict = "cuVS somewhat faster; hipVS in the same ballpark (peer GPUs)."
        else:
            verdict = "cuVS clearly faster on this suite; hipVS not yet competitive on wall time."
        print(f"\nManager line: {verdict}")
        print(
            f"  (geo-mean case time: hipVS is {geo:.2f}× cuVS; "
            f"<1 = hip faster, ~1 = parity, >1 = cu faster)"
        )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out_json}")

    if args.out_md:
        lines = [
            "# Shared gtest timing — hipVS vs cuVS",
            "",
            f"- hipVS: **{h.get('n_pass')}/{h.get('n_cases')}** pass, "
            f"sum case ms={h.get('sum_case_ms')}, GPU=`{h.get('gpu_name')}`",
            f"- cuVS: **{c.get('n_pass')}/{c.get('n_cases')}** pass, "
            f"sum case ms={c.get('sum_case_ms')}, GPU=`{c.get('gpu_name')}`",
            f"- Common timed (both PASS): **{len(ratios)}**",
            "",
        ]
        if med is not None:
            lines += [
                f"- Median ratio hip/cu: **{med:.3f}×**",
                f"- Geo-mean ratio hip/cu: **{geo:.3f}×**",
                f"- Sum(ms) ratio hip/cu: **{summary['sum_ms_ratio_hip_over_cu']:.3f}×**",
                "",
                f"**Manager line:** {verdict}",
                "",
            ]
        lines += [
            "Ratio = hipVS_ms / cuVS_ms. Peer GPUs (not same silicon).",
            "Unit tests are correctness-sized — not production QPS.",
            "",
        ]
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {args.out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
