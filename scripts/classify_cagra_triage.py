#!/usr/bin/env python3
"""Classify Phase A CAGRA triage from lib JSON + optional Catch2 log greps.

Usage:
  python3 scripts/classify_cagra_triage.py \\
    --hipvs-json path/to/lib_hipvs_cagra.json \\
    [--catch2-log path/to/cagra_catch2.log]

Exit 0 always; prints recommended ownership (hipVS vs Knowhere).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def max_recall(doc: dict) -> float:
    best = -1.0
    for key in ("itopk_results", "nprobe_results"):
        for row in doc.get(key) or []:
            for k, v in row.items():
                if k.startswith("recall@"):
                    best = max(best, float(v))
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hipvs-json", required=True)
    p.add_argument("--catch2-log", default="")
    p.add_argument("--recall-ok", type=float, default=0.5)
    args = p.parse_args()

    doc = json.loads(Path(args.hipvs_json).read_text(encoding="utf-8"))
    r = max_recall(doc)
    print(f"hipVS lib max recall@k = {r:.4f} (threshold {args.recall_ok})")

    catch2_cagra_fail = None
    if args.catch2_log:
        text = Path(args.catch2_log).read_text(encoding="utf-8", errors="replace")
        if re.search(r"CAGRA", text, re.I):
            catch2_cagra_fail = bool(
                re.search(r"FAILED|failed \d|0 failed", text)
                and re.search(r"0\.0|recall", text, re.I)
            ) or ("failed" in text.lower() and "passed" in text.lower())
            # Heuristic: any FAILED near CAGRA
            catch2_cagra_fail = bool(re.search(r"FAILED", text))
        print(f"Catch2 log mentions CAGRA; FAILED present={catch2_cagra_fail}")

    print()
    if r < 0.05:
        print("OWNER: hipVS / ROCm-DS (library recall ~0 on gfx1100)")
        print("ACTION: minimal repro upstream; check USE_WARPSIZE_32 / graph build")
        print("BLOCK Phase B until hipVS search returns non-zero recall.")
    elif r >= args.recall_ok:
        print("OWNER: Knowhere HIP wiring / serialize (hipVS lib looks OK)")
        print("ACTION: patch Knowhere GPU_CUVS_CAGRA path; land under patches/knowhere/")
        if catch2_cagra_fail:
            print("Catch2 still red — focus TopK + Serialize/Deserialize sections.")
        print("Phase B only after Catch2 CAGRA green (or Serialize fixed for load).")
    else:
        print("OWNER: mixed / quality gap (lib recall mid-range)")
        print("ACTION: tune graph_degree / itopk; compare to cuVS peer JSON")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
