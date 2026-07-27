#!/usr/bin/env python3
"""Compare two Milvus Layer-4 JSON results (AMD vs CUDA).

Speed-up = AMD QPS / CUDA QPS. Recall must match within --recall-tol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def by_nprobe(doc: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in doc.get("nprobe_results", []):
        out[int(row["nprobe"])] = row
    return out


def recall_key(row: dict) -> str:
    for k in row:
        if k.startswith("recall@"):
            return k
    return "recall@10"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--amd", required=True, type=Path, help="AMD HIP Milvus JSON")
    ap.add_argument("--cuda", required=True, type=Path, help="CUDA Milvus JSON")
    ap.add_argument(
        "--recall-tol",
        type=float,
        default=0.01,
        help="Max |AMD-CUDA| recall delta to treat as match (default 0.01)",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional markdown summary path",
    )
    args = ap.parse_args()

    amd = load(args.amd)
    cuda = load(args.cuda)
    a = by_nprobe(amd)
    c = by_nprobe(cuda)
    nprobes = sorted(set(a) & set(c))
    if not nprobes:
        print("ERROR: no overlapping nprobe values", file=sys.stderr)
        return 1

    rk = recall_key(a[nprobes[0]])
    lines = [
        "# Milvus AMD vs CUDA — GIST GPU_IVF_PQ",
        "",
        f"AMD: `{args.amd}`",
        f"CUDA: `{args.cuda}`",
        f"Index: {amd.get('index_type')} m={amd.get('m')} nbits={amd.get('nbits')} "
        f"nlist={amd.get('nlist')}",
        f"Data: {amd.get('data_path')} xb={amd.get('xb_shape')} xq={amd.get('xq_shape')}",
        "",
        f"| nprobe | AMD QPS | CUDA QPS | {rk} AMD | {rk} CUDA | AMD/CUDA | recallΔ |",
        "|--------|---------|----------|----------|-----------|----------|---------|",
    ]
    print(
        f"{'nprobe':>6}  {'AMD_QPS':>10}  {'CUDA_QPS':>10}  "
        f"{'R_AMD':>7}  {'R_CUDA':>7}  {'AMD/CUDA':>8}  {'dR':>7}"
    )
    ok = True
    for npv in nprobes:
        aq = float(a[npv]["qps"])
        cq = float(c[npv]["qps"])
        ra = float(a[npv].get(rk, float("nan")))
        rc = float(c[npv].get(rk, float("nan")))
        ratio = aq / cq if cq > 0 else float("nan")
        dr = abs(ra - rc)
        if dr > args.recall_tol:
            ok = False
        flag = "" if dr <= args.recall_tol else "  !! recall"
        print(
            f"{npv:6d}  {aq:10.1f}  {cq:10.1f}  "
            f"{ra:7.4f}  {rc:7.4f}  {ratio:7.2f}x  {dr:7.4f}{flag}"
        )
        lines.append(
            f"| {npv} | {aq:.1f} | {cq:.1f} | {ra:.4f} | {rc:.4f} | "
            f"**{ratio:.2f}×** | {dr:.4f} |"
        )

    lines.extend(
        [
            "",
            "Speed-up = AMD QPS / CUDA QPS. Target for GPU-heavy story: AMD/CUDA ≤ ~0.85× "
            "at matched recall (NVIDIA ahead).",
            "",
        ]
    )
    text = "\n".join(lines) + "\n"
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(text, encoding="utf-8")
        print(f"wrote {args.out_md}")
    if not ok:
        print(
            f"WARNING: recall delta > {args.recall_tol} on at least one nprobe "
            "(do not quote speed until fixed)",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
