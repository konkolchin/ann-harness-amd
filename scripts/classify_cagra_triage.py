#!/usr/bin/env python3
"""Classify Phase A CAGRA triage from lib JSON + optional Catch2 / hipVS logs.

Usage:
  python3 scripts/classify_cagra_triage.py \\
    --hipvs-json "$WORKDIR/logs/lib_hipvs_cagra_*.json" \\
    --hipvs-log "$WORKDIR/logs/cagra_hipvs_minimal_*.log" \\
    --catch2-log "$WORKDIR/logs/cagra_catch2_*.log"

Bash may expand globs into many words — pass a quoted glob, or pass many paths;
both work (newest file wins per flag).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path


def resolve_one(patterns: list[str] | str | None, label: str) -> Path | None:
    if not patterns:
        return None
    if isinstance(patterns, str):
        patterns = [patterns]
    matches: list[str] = []
    for pattern in patterns:
        if not pattern:
            continue
        hit = sorted(glob.glob(pattern))
        if hit:
            matches.extend(hit)
        else:
            p = Path(pattern)
            if p.is_file():
                matches.append(str(p))
    matches = sorted(set(matches))
    if not matches:
        print(f"NOTE: no file for {label}={patterns!r}", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(
            f"NOTE: {label} matched {len(matches)} files; using newest {matches[-1]}",
            file=sys.stderr,
        )
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
        nargs="*",
        default=[],
        help="Lib bench JSON (glob or multiple paths OK)",
    )
    p.add_argument(
        "--hipvs-log",
        nargs="*",
        default=[],
        help="hipVS bench stdout log(s) — used when build throws before JSON",
    )
    p.add_argument(
        "--catch2-log",
        nargs="*",
        default=[],
        help="Catch2 log (glob or multiple paths OK)",
    )
    p.add_argument("--recall-ok", type=float, default=0.5)
    args = p.parse_args()

    hipvs_path = resolve_one(args.hipvs_json, "hipvs-json")
    hipvs_log = resolve_one(args.hipvs_log, "hipvs-log")
    catch2_path = resolve_one(args.catch2_log, "catch2-log")

    if hipvs_path is None and catch2_path is None and hipvs_log is None:
        raise SystemExit(
            "Need at least one of --hipvs-json, --hipvs-log, or --catch2-log.\n"
            "Quote globs: --hipvs-log \"$WORKDIR/logs/cagra_hipvs_*.log\""
        )

    if hipvs_log is not None:
        htext = hipvs_log.read_text(encoding="utf-8", errors="replace")
        print(f"hipVS log: {hipvs_log}")
        if "using ivf_pq::index_params" in htext and "NN_DESCENT" in htext:
            print(
                "NOTE: log requested NN_DESCENT but still used IVF_PQ "
                "(bindings likely dropped graph_build_algo — pull latest bench_cuvs_cagra.py)."
            )
        if "invalid or duplicated neighbor" in htext or "norm computation" in htext:
            print()
            print("OWNER: hipVS / ROCm-DS (CAGRA intermediate knn graph on gfx1100)")
            print(
                "EVIDENCE: RAFT graph_core — too many invalid/duplicated neighbors "
                "(IVF_PQ intermediate path if 'using ivf_pq::index_params' in log)."
            )
            print("ACTION:")
            print("  1) git pull; re-run with GRAPH_BUILD_ALGO=NN_DESCENT")
            print("     (bench must print verify IndexParams.graph_build_algo=...)")
            print("  2) If still IVF_PQ or still throws → escalate to ROCm-DS")
            print("  3) BLOCK Phase B until library build/search works")
            print(
                "NOTE: Knowhere Catch2 recall 0.0 is consistent with broken hipVS CAGRA."
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
        print("hipVS json: missing")

    catch2_failed = None
    if catch2_path is not None:
        text = catch2_path.read_text(encoding="utf-8", errors="replace")
        catch2_cagra_mentions = bool(re.search(r"CAGRA", text, re.I))
        catch2_failed = bool(re.search(r"\bFAILED\b|\bfailed\b", text))
        if re.search(r"Unrecognised token|Error\(s\) in input", text):
            print(
                f"Catch2 log: {catch2_path}\n"
                "  STATUS: filter/CLI error (not a suite result)."
            )
            catch2_failed = None
        else:
            print(
                f"Catch2 log: {catch2_path}\n"
                f"  CAGRA mentioned={catch2_cagra_mentions} FAILED present={catch2_failed}"
            )

    print()
    if r is None:
        print("OWNER: unresolved without successful hipVS JSON — see hipVS log OWNER above if any")
        print("ACTION: fix graph_build_algo application or escalate RAFT graph_core error")
        if catch2_failed:
            print("Catch2 still red — consistent with CAGRA recall 0.0 baseline.")
    elif r < 0.05:
        print("OWNER: hipVS / ROCm-DS (library recall ~0 on gfx1100)")
        print("ACTION: minimal repro upstream; BLOCK Phase B")
    elif r >= args.recall_ok:
        print("OWNER: Knowhere HIP wiring / serialize (hipVS lib looks OK)")
        print("ACTION: patch Knowhere GPU_CUVS_CAGRA; land under patches/knowhere/")
        if catch2_failed:
            print("Catch2 still red — focus TopK + Serialize/Deserialize.")
    else:
        print("OWNER: mixed / quality gap (lib recall mid-range)")
        print("ACTION: tune graph_degree / itopk; compare to cuVS peer JSON")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
