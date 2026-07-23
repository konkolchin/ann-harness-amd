#!/usr/bin/env python3
"""Run shared cuVS/hipVS gtest binaries and capture per-test wall times.

Google Test already prints ``(N ms)`` per case. This wrapper runs a fixed list of
binaries (same names on hipVS and cuVS trees), parses pass/fail + ms, and writes
JSON for a peer-GPU compare.

Manager framing (not a microbench recipe):
  RDNA3 hipVS (warp_size=32) vs NVIDIA cuVS — same unit tests both libraries ship.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

# Default suite ≈ Layer-1 neighbor correctness (100+ parameterized cases across
# these binaries). Override with --binaries or GTEST_BINARIES=a;b;c
# Manager compare needs ~100 shared cases; float IVF alone is 98/98 on gfx1100.
# Extra binaries are optional — skip silently if not built (--limit-tests).
DEFAULT_BINARIES = [
    "NEIGHBORS_ANN_IVF_FLAT_TEST",
    "NEIGHBORS_ANN_IVF_PQ_TEST",
    "NEIGHBORS_ANN_BRUTE_FORCE_TEST",
]

# Optional gtest filters (empty = all cases in that binary)
DEFAULT_FILTERS = {
    # Float IVF packer/search path — the 98/98 gfx1100 story
    "NEIGHBORS_ANN_IVF_FLAT_TEST": "AnnIVFFlatTest/AnnIVFFlatTestF_float.*",
}

OK_RE = re.compile(
    r"^\[\s+(OK|FAILED)\s+\]\s+(\S+)\s+\((\d+)\s+ms\)\s*$"
)
SUMMARY_RE = re.compile(
    r"^\[\s*(PASSED|FAILED)\s*\]\s+(\d+)\s+tests?", re.IGNORECASE
)


def detect_gpu(backend: str) -> str:
    if backend == "hipvs":
        try:
            out = subprocess.check_output(
                ["rocminfo"], text=True, stderr=subprocess.DEVNULL, timeout=30
            )
            for line in out.splitlines():
                if "gfx" in line.lower() and "Name:" in line:
                    return line.strip()
                if "Marketing Name" in line or "GFX Version" in line:
                    return line.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        return os.environ.get("HIP_VISIBLE_DEVICES", "rocm-unknown")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return out.strip().splitlines()[0].strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "cuda-unknown"


def parse_gtest_log(text: str) -> list[dict]:
    cases: list[dict] = []
    for line in text.splitlines():
        m = OK_RE.match(line.strip())
        if not m:
            continue
        status, name, ms = m.group(1), m.group(2), int(m.group(3))
        cases.append(
            {
                "name": name,
                "status": "PASS" if status == "OK" else "FAIL",
                "ms": ms,
            }
        )
    return cases


def run_one(
    binary: Path,
    gtest_filter: str | None,
    extra_env: dict[str, str],
) -> tuple[list[dict], float, int, str]:
    cmd = [str(binary), "--gtest_print_time=1", "--gtest_color=no"]
    if gtest_filter:
        cmd.append(f"--gtest_filter={gtest_filter}")

    env = os.environ.copy()
    env.update(extra_env)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(os.environ.get("GTEST_TIMEOUT_S", "7200")),
    )
    wall_s = time.perf_counter() - t0
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    cases = parse_gtest_log(log)
    return cases, wall_s, proc.returncode, log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend",
        choices=("hipvs", "cuvs"),
        required=True,
        help="Label only; binaries come from --gtest-dir",
    )
    ap.add_argument(
        "--gtest-dir",
        type=Path,
        required=True,
        help="Directory with NEIGHBORS_*_TEST binaries (cpp/build/gtests)",
    )
    ap.add_argument(
        "--binaries",
        default=",".join(DEFAULT_BINARIES),
        help="Comma-separated binary names under --gtest-dir",
    )
    ap.add_argument(
        "--filters-json",
        default="",
        help='Optional JSON map binary->gtest_filter, e.g. \'{"NEIGHBORS_ANN_IVF_FLAT_TEST":"*"}\'',
    )
    ap.add_argument(
        "--no-default-filters",
        action="store_true",
        help="Run full binaries (ignore DEFAULT_FILTERS)",
    )
    ap.add_argument("--results-json", type=Path, required=True)
    ap.add_argument(
        "--save-logs-dir",
        type=Path,
        default=None,
        help="If set, write raw gtest stdout/stderr per binary",
    )
    args = ap.parse_args()

    gtest_dir = args.gtest_dir.expanduser().resolve()
    if not gtest_dir.is_dir():
        raise SystemExit(f"gtest dir not found: {gtest_dir}")

    filters = {} if args.no_default_filters else dict(DEFAULT_FILTERS)
    if args.filters_json:
        filters.update(json.loads(args.filters_json))

    binaries = [b.strip() for b in args.binaries.split(",") if b.strip()]
    gpu = detect_gpu(args.backend)
    host = platform.node()

    suite: list[dict] = []
    all_cases: list[dict] = []
    t_suite0 = time.perf_counter()

    skip_missing = os.environ.get("GTEST_SKIP_MISSING", "1") != "0"
    for name in binaries:
        path = gtest_dir / name
        if not path.is_file():
            msg = f"missing binary: {path}"
            if skip_missing and name != binaries[0]:
                print(f"==> skip {name} ({msg})", flush=True)
                continue
            if skip_missing and name == binaries[0]:
                raise SystemExit(
                    f"{msg}\n"
                    "Rebuild hipVS/cuVS with tests, e.g.\n"
                    "  ./build.sh libcuvs tests --limit-tests=NEIGHBORS_ANN_IVF_FLAT_TEST"
                )
            suite.append(
                {
                    "binary": name,
                    "error": msg,
                    "cases": [],
                    "wall_s": 0.0,
                    "exit_code": 127,
                }
            )
            continue
        filt = filters.get(name)
        print(f"==> {name}" + (f"  filter={filt}" if filt else "  (all cases)"), flush=True)
        cases, wall_s, rc, log = run_one(path, filt, {})
        if args.save_logs_dir:
            args.save_logs_dir.mkdir(parents=True, exist_ok=True)
            (args.save_logs_dir / f"{name}.log").write_text(log, encoding="utf-8")
        n_pass = sum(1 for c in cases if c["status"] == "PASS")
        n_fail = sum(1 for c in cases if c["status"] == "FAIL")
        print(
            f"    cases={len(cases)} pass={n_pass} fail={n_fail} "
            f"wall={wall_s:.1f}s exit={rc}",
            flush=True,
        )
        for c in cases:
            c2 = dict(c)
            c2["binary"] = name
            all_cases.append(c2)
        suite.append(
            {
                "binary": name,
                "filter": filt,
                "cases": cases,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "wall_s": wall_s,
                "exit_code": rc,
                "sum_case_ms": sum(c["ms"] for c in cases),
            }
        )

    suite_wall_s = time.perf_counter() - t_suite0
    n_pass = sum(1 for c in all_cases if c["status"] == "PASS")
    n_fail = sum(1 for c in all_cases if c["status"] == "FAIL")
    sum_ms = sum(c["ms"] for c in all_cases)

    payload = {
        "protocol": "shared_gtest_wallclock",
        "backend": args.backend,
        "gpu_name": gpu,
        "host": host,
        "gtest_dir": str(gtest_dir),
        "binaries": binaries,
        "filters": filters,
        "suite_wall_s": suite_wall_s,
        "n_cases": len(all_cases),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "sum_case_ms": sum_ms,
        "suite": suite,
        "cases": all_cases,
    }
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.results_json}")
    print(f"  cases={len(all_cases)} pass={n_pass} fail={n_fail} sum_ms={sum_ms} suite_s={suite_wall_s:.1f}")
    return 0 if n_fail == 0 and all(s.get("exit_code", 1) == 0 for s in suite) else 1


if __name__ == "__main__":
    raise SystemExit(main())
