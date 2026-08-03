#!/usr/bin/env python3
"""Classify Phase A CAGRA triage from lib JSON + optional Catch2 log.

Usage:
  python3 scripts/classify_cagra_triage.py \\
    --hipvs-json path/to/lib_hipvs_cagra.json \\
    --catch2-log path/to/cagra_catch2.log

  # Globs OK (expanded in-process):
  python3 scripts/classify_cagra_triage.py \\
    --hipvs-json '$WORKDIR/logs/lib_hipvs_cagra_minimal_*.json' \\
    --catch2-log '$WORKDIR/logs/cagra_catch2_*.log'

  # Catch2-only if hipVS Python not ready yet:
  python3 scripts/classify_cagra_triage.py --catch2-log path/to/cagra_catch2.log
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path


def resolve_one(pattern: str, label: str) -> Path | None:
    if not pattern:
        return None
    matches = sorted(glob.glob(pattern))
    if not matches:
        # literal path without metachar
        p = Path(pattern)
        if p.is_file():
            return p
        print(f"NOTE: no file for {label}={pattern!r}", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(f"NOTE: {label} matched {len(matches)} files; using newest {matches[-1]}", file=sys.stderr)
    return Path(matches[-1])


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
    p.add_argument(
        "--hipvs-json",
        default="",
        help="Lib bench JSON (glob OK). Optional if only Catch2 available.",
    )
    p.add_argument(
        "--hipvs-log",
        default="",
        help="hipVS bench stdout log (glob OK) — used when build throws before JSON",
    )
    p.add_argument("--catch2-log", default="", help="Catch2 log (glob OK)")
    p.add_argument("--recall-ok", type=float, default=0.5)
    args = p.parse_args()

    hipvs_path = resolve_one(args.hipvs_json, "hipvs-json")
    hipvs_log = resolve_one(args.hipvs_log, "hipvs-log")
    catch2_path = resolve_one(args.catch2_log, "catch2-log")

    if hipvs_path is None and catch2_path is None and hipvs_log is None:
        raise SystemExit(
            "Need at least one of --hipvs-json, --hipvs-log, or --catch2-log.\n"
            "If hipVS Python lacks cagra: source ~/hipvs-bench-venv/bin/activate and re-run reproduce."
        )

    if hipvs_log is not None:
        htext = hipvs_log.read_text(encoding="utf-8", errors="replace")
        print(f"hipVS log: {hipvs_log}")
        if "invalid or duplicated neighbor" in htext or "norm computation" in htext:
            print()
            print("OWNER: hipVS / ROCm-DS (CAGRA intermediate knn graph on gfx1100)")
            print(
                "EVIDENCE: RAFT graph_core — too many invalid/duplicated neighbors "
                "(often IVF_PQ path / norm overflow)."
            )
            print("ACTION:")
            print("  1) Retry: GRAPH_BUILD_ALGO=NN_DESCENT bash scripts/run_hipvs_cagra_bench.sh")
            print("  2) If still broken: minimal repro → ROCm-DS/hipVS (consumer gfx1100)")
            print("  3) BLOCK Phase B (Milvus GPU_CAGRA) until library build/search works")
            print(
                "NOTE: Knowhere Catch2 recall 0.0 is consistent with broken hipVS CAGRA "
                "(build may not throw; search returns garbage)."
            )
            return
        if "CAGRA build FAILED" in htext or "CuvsException" in htext:
            print("hipVS CAGRA build raised an exception (see log); treat as library ownership.")

    r = None
    if hipvs_path is not None:
        doc = json.loads(hipvs_path.read_text(encoding="utf-8"))
        r = max_recall(doc)
        print(f"hipVS json: {hipvs_path}")
        print(f"hipVS lib max recall@k = {r:.4f} (threshold {args.recall_ok})")
    else:
        print("hipVS json: missing — classify from Catch2 / install hints only")

    catch2_failed = None
    catch2_cagra_mentions = False
    if catch2_path is not None:
        text = catch2_path.read_text(encoding="utf-8", errors="replace")
        catch2_cagra_mentions = bool(re.search(r"CAGRA", text, re.I))
        catch2_failed = bool(re.search(r"\bFAILED\b|\bfailed\b", text))
        # usage / filter errors are not a real suite result
        if re.search(r"Unrecognised token|Error\(s\) in input", text):
            print(
                f"Catch2 log: {catch2_path}\n"
                "  STATUS: filter/CLI error (not a suite result). "
                "Re-run reproduce_cagra_gfx1100.sh (uses \"Test All GPU Index\")."
            )
            catch2_failed = None
        else:
            print(
                f"Catch2 log: {catch2_path}\n"
                f"  CAGRA mentioned={catch2_cagra_mentions} FAILED/failed present={catch2_failed}"
            )
            for line in text.splitlines():
                if re.search(r"CAGRA|failed|FAILED|assertions", line, re.I):
                    if "CAGRA" in line.upper() or "failed" in line.lower() or "FAILED" in line:
                        print(f"  | {line[:160]}")

    print()
    if r is None:
        print("OWNER: unresolved — need hipVS Python cagra for library split")
        print("ACTION:")
        print("  1) source ~/hipvs-bench-venv/bin/activate")
        print("  2) python3 -c \"from cuvs.neighbors import cagra; print('ok')\"")
        print("  3) if import fails: docs/hipvs_vs_cuvs_bench.md §1 (build hipVS python)")
        print("  4) SKIP_CATCH2=1 bash scripts/reproduce_cagra_gfx1100.sh")
        if catch2_failed:
            print("Catch2 still shows failures — consistent with 2026-07-26 CAGRA recall 0.0 baseline.")
    elif r < 0.05:
        print("OWNER: hipVS / ROCm-DS (library recall ~0 on gfx1100)")
        print("ACTION: minimal repro upstream; check USE_WARPSIZE_32 / graph build")
        print("BLOCK Phase B until hipVS search returns non-zero recall.")
    elif r >= args.recall_ok:
        print("OWNER: Knowhere HIP wiring / serialize (hipVS lib looks OK)")
        print("ACTION: patch Knowhere GPU_CUVS_CAGRA path; land under patches/knowhere/")
        if catch2_failed:
            print("Catch2 still red — focus TopK + Serialize/Deserialize sections.")
        print("Phase B only after Catch2 CAGRA green (or Serialize fixed for load).")
    else:
        print("OWNER: mixed / quality gap (lib recall mid-range)")
        print("ACTION: tune graph_degree / itopk; compare to cuVS peer JSON")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
