#!/usr/bin/env python3
"""Library-level IVF Flat / IVF-PQ bench via the cuVS Python API.

Works on:
  - NVIDIA: RAPIDS / pip cuVS + CuPy (CUDA)
  - AMD:    hipVS Python build (import still named ``cuvs``) + CuPy (ROCm)

No Milvus / Knowhere. Same SIFT-1M recipe as Layer-4 for fair hipVS vs cuVS QPS.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

DEFAULT_DATA_PATH = "data/sift-128-euclidean.hdf5"
DEFAULT_K = 10
DEFAULT_NLIST = 1024
DEFAULT_NPROBES = "1,4,8,16,32"
DEFAULT_P99_SAMPLE = 500
DEFAULT_WARMUP = 1


def recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    hits = 0
    nq = pred_ids.shape[0]
    for i in range(nq):
        hits += len(set(pred_ids[i, :k]).intersection(set(gt_ids[i, :k])))
    return hits / (nq * k)


def detect_backend() -> tuple[str, str]:
    """Return (backend_tag, gpu_name). Prefer nvidia-smi, then rocm-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        if out:
            return "cuvs", out.splitlines()[0].strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        for line in out.splitlines():
            if "Card series" in line or "Card model" in line or "GFX" in line.upper():
                return "hipvs", line.split(":", 1)[-1].strip()
        return "hipvs", "AMD GPU (rocm-smi)"
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return "unknown", "unknown"


def sync_resources(resources) -> None:
    if resources is not None and hasattr(resources, "sync"):
        resources.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description="hipVS/cuVS library IVF microbench")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="sift-128-euclidean.hdf5")
    parser.add_argument(
        "--index-type",
        default="IVF_FLAT",
        choices=["IVF_FLAT", "IVF_PQ"],
        help="Library index (not Milvus GPU_*)",
    )
    parser.add_argument("--nlist", type=int, default=DEFAULT_NLIST)
    parser.add_argument("--nprobes", default=DEFAULT_NPROBES)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--m",
        type=int,
        default=32,
        help="PQ subspaces (Milvus m). Mapped to cuVS ivf_pq IndexParams.pq_dim",
    )
    parser.add_argument("--nbits", type=int, default=8, help="PQ bits (cuVS pq_bits)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--p99-sample", type=int, default=DEFAULT_P99_SAMPLE)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-query-rows", type=int, default=0)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "hipvs", "cuvs"],
        help="Tag written into JSON (auto = detect via nvidia-smi / rocm-smi)",
    )
    parser.add_argument("--results-json", default="", help="Write metrics JSON here")
    args = parser.parse_args()

    nprobes = [int(x.strip()) for x in args.nprobes.split(",") if x.strip()]

    try:
        import cupy as cp
        from cuvs.neighbors import ivf_flat, ivf_pq
    except ImportError as exc:
        raise SystemExit(
            "Need cupy + cuvs Python packages.\n"
            "  NVIDIA: conda/pip RAPIDS cuVS\n"
            "  AMD:    build hipVS Python (import still named cuvs)\n"
            f"Import error: {exc}"
        ) from exc

    try:
        from cuvs.common import Resources

        resources = Resources()
    except ImportError:
        resources = None

    detected_tag, gpu_name = detect_backend()
    backend = detected_tag if args.backend == "auto" else args.backend

    with h5py.File(args.data, "r") as f:
        xb = np.asarray(f["train"], dtype=np.float32)
        xq = np.asarray(f["test"], dtype=np.float32)
        gt = np.asarray(f["neighbors"], dtype=np.int64)

    if args.max_train_rows > 0:
        xb = xb[: args.max_train_rows]
    if args.max_query_rows > 0:
        xq = xq[: args.max_query_rows]
        gt = gt[: args.max_query_rows]

    dim = int(xb.shape[1])
    if args.index_type == "IVF_PQ" and dim % args.m != 0:
        raise SystemExit(f"PQ m={args.m} must divide dim={dim}")

    print(f"backend={backend} gpu={gpu_name!r}")
    print(f"index_type={args.index_type} nlist={args.nlist} nprobes={nprobes} k={args.k}")
    if args.index_type == "IVF_PQ":
        print(f"pq_params: m/pq_dim={args.m} nbits/pq_bits={args.nbits}")
    print(f"xb={xb.shape} xq={xq.shape} gt={gt.shape} dim={dim}")

    xb_g = cp.asarray(xb)
    xq_g = cp.asarray(xq)
    sync_resources(resources)
    if resources is None:
        cp.cuda.Device().synchronize()

    def _sync() -> None:
        sync_resources(resources)
        if resources is None:
            cp.cuda.Device().synchronize()

    build_kw = {"resources": resources} if resources is not None else {}
    search_kw = dict(build_kw)

    t0 = time.perf_counter()
    if args.index_type == "IVF_FLAT":
        build_params = ivf_flat.IndexParams(n_lists=args.nlist, metric="sqeuclidean")
        index = ivf_flat.build(build_params, xb_g, **build_kw)
        search_mod = ivf_flat
    else:
        build_params = ivf_pq.IndexParams(
            n_lists=args.nlist,
            metric="sqeuclidean",
            pq_dim=args.m,
            pq_bits=args.nbits,
        )
        index = ivf_pq.build(build_params, xb_g, **build_kw)
        search_mod = ivf_pq
    _sync()
    build_s = time.perf_counter() - t0
    print(f"index_build_time_s={build_s:.2f}")

    results = {
        "protocol": "library_cuvs_api",
        "backend": backend,
        "gpu_name": gpu_name,
        "host": platform.node(),
        "index_type": args.index_type,
        "nlist": args.nlist,
        "m": args.m if args.index_type == "IVF_PQ" else None,
        "nbits": args.nbits if args.index_type == "IVF_PQ" else None,
        "k": args.k,
        "nprobes": nprobes,
        "xb_shape": list(xb.shape),
        "xq_shape": list(xq.shape),
        "data_path": args.data,
        "timings_s": {"index_build": build_s},
        "nprobe_results": [],
        "cuvs": getattr(__import__("cuvs"), "__version__", "unknown"),
    }

    print(f"\n{backend} {args.index_type} results:")
    for nprobe in nprobes:
        search_params = search_mod.SearchParams(n_probes=nprobe)

        for _ in range(max(0, args.warmup)):
            distances, neighbors = search_mod.search(
                search_params, index, xq_g, args.k, **search_kw
            )
            _sync()

        t0 = time.perf_counter()
        distances, neighbors = search_mod.search(
            search_params, index, xq_g, args.k, **search_kw
        )
        _sync()
        elapsed = time.perf_counter() - t0
        qps = xq.shape[0] / elapsed

        pred = cp.asnumpy(neighbors).astype(np.int64)
        r = recall_at_k(pred, gt, args.k)

        lat_ms: list[float] = []
        sample_n = min(args.p99_sample, xq.shape[0])
        for i in range(sample_n):
            q = xq_g[i : i + 1]
            s0 = time.perf_counter()
            search_mod.search(search_params, index, q, args.k, **search_kw)
            _sync()
            lat_ms.append((time.perf_counter() - s0) * 1000.0)
        p99 = float(np.percentile(lat_ms, 99)) if lat_ms else float("nan")

        print(f"nprobe={nprobe:2d} qps={qps:8.1f} p99_ms={p99:7.2f} recall@{args.k}={r:.4f}")
        results["nprobe_results"].append(
            {
                "nprobe": nprobe,
                "qps": qps,
                f"recall@{args.k}": r,
                "p99_ms": p99,
                "batch_search_s": elapsed,
            }
        )

    if args.results_json:
        out = Path(args.results_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"results_json={out}")


if __name__ == "__main__":
    main()
