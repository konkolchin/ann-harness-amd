#!/usr/bin/env python3
"""Library-level CAGRA bench via the cuVS / hipVS Python API.

Works on:
  - NVIDIA: RAPIDS / pip cuVS + CuPy (CUDA)
  - AMD:    hipVS Python build (import still named ``cuvs``) + CuPy (ROCm)

No Milvus / Knowhere. Companion to bench_cuvs_ivf.py for the CAGRA follow-on.
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
DEFAULT_ITOPK = "32,64,128,256"
DEFAULT_GRAPH_DEGREE = 32
DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 64
DEFAULT_P99_SAMPLE = 0
DEFAULT_WARMUP = 1


def recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    hits = 0
    nq = pred_ids.shape[0]
    for i in range(nq):
        hits += len(set(pred_ids[i, :k]).intersection(set(gt_ids[i, :k])))
    return hits / (nq * k)


def detect_backend() -> tuple[str, str]:
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


def device_ids_to_numpy(neighbors, cp_mod) -> np.ndarray:
    if hasattr(neighbors, "__cuda_array_interface__"):
        return np.asarray(cp_mod.asarray(neighbors).get(), dtype=np.int64)
    if hasattr(neighbors, "copy_to_host"):
        return np.asarray(neighbors.copy_to_host(), dtype=np.int64)
    return np.asarray(neighbors).astype(np.int64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="hipVS/cuVS library CAGRA microbench")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--itopk-sizes",
        default=DEFAULT_ITOPK,
        help="Comma-separated CAGRA itopk_size search sweep",
    )
    parser.add_argument("--graph-degree", type=int, default=DEFAULT_GRAPH_DEGREE)
    parser.add_argument(
        "--intermediate-graph-degree",
        type=int,
        default=DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
    )
    parser.add_argument(
        "--search-width",
        type=int,
        default=1,
        help="CAGRA SearchParams.search_width",
    )
    parser.add_argument(
        "--graph-build-algo",
        default="",
        help="Optional CAGRA IndexParams.graph_build_algo: IVF_PQ | NN_DESCENT "
        "(empty = library default; on gfx1100 IVF_PQ intermediate graph often fails)",
    )
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        help="L2-normalize train/query rows before build/search (can reduce norm overflow)",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--p99-sample", type=int, default=DEFAULT_P99_SAMPLE)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-query-rows", type=int, default=0)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "hipvs", "cuvs"],
    )
    parser.add_argument("--results-json", default="")
    args = parser.parse_args()

    itopk_sizes = [int(x.strip()) for x in args.itopk_sizes.split(",") if x.strip()]

    try:
        import cupy as cp
        from cuvs.neighbors import cagra
    except ImportError as exc:
        raise SystemExit(
            "Need cupy + cuvs Python packages with neighbors.cagra importable.\n"
            "  NVIDIA: pip cuVS + CuPy CUDA\n"
            "  AMD: hipVS Python (import still named cuvs) — see docs/cagra_consumer_followon.md\n"
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

    if args.l2_normalize:
        def _l2(a: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(a, axis=1, keepdims=True)
            n = np.maximum(n, 1e-12)
            return (a / n).astype(np.float32)

        xb = _l2(xb)
        xq = _l2(xq)
        print("l2_normalize=1 (note: gt still Euclidean on original SIFT — recall may drop)")

    dim = int(xb.shape[1])
    print(f"backend={backend} gpu={gpu_name!r}")
    print(
        f"index_type=CAGRA graph_degree={args.graph_degree} "
        f"intermediate_graph_degree={args.intermediate_graph_degree}"
    )
    print(
        f"itopk_sizes={itopk_sizes} search_width={args.search_width} k={args.k} "
        f"graph_build_algo={args.graph_build_algo or 'default'}"
    )
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

    build_params_kw: dict = {
        "metric": "sqeuclidean",
        "graph_degree": args.graph_degree,
        "intermediate_graph_degree": args.intermediate_graph_degree,
    }
    if args.graph_build_algo:
        # cuVS Python: graph_build_algo="NN_DESCENT" | "IVF_PQ"
        build_params_kw["graph_build_algo"] = args.graph_build_algo

    def _make_params(kw: dict):
        try:
            return cagra.IndexParams(**kw)
        except TypeError:
            kw2 = dict(kw)
            kw2.pop("intermediate_graph_degree", None)
            try:
                return cagra.IndexParams(**kw2)
            except TypeError:
                kw2.pop("graph_build_algo", None)
                return cagra.IndexParams(**kw2)

    build_params = _make_params(build_params_kw)

    t0 = time.perf_counter()
    try:
        index = cagra.build(build_params, xb_g, **build_kw)
    except Exception as exc:  # noqa: BLE001 - surface library ownership clearly
        msg = str(exc)
        print(f"CAGRA build FAILED: {msg.splitlines()[0][:300]}")
        if "invalid or duplicated neighbor" in msg or "norm computation" in msg:
            print(
                "OWNER HINT: hipVS/cuVS CAGRA intermediate knn graph on this GPU "
                "(often IVF_PQ path). Try --graph-build-algo NN_DESCENT or escalate to ROCm-DS."
            )
        raise
    _sync()
    build_s = time.perf_counter() - t0
    print(f"index_build_time_s={build_s:.2f}")

    results = {
        "protocol": "library_cuvs_api",
        "backend": backend,
        "gpu_name": gpu_name,
        "host": platform.node(),
        "index_type": "CAGRA",
        "graph_degree": args.graph_degree,
        "intermediate_graph_degree": args.intermediate_graph_degree,
        "search_width": args.search_width,
        "graph_build_algo": args.graph_build_algo or "default",
        "l2_normalize": bool(args.l2_normalize),
        "k": args.k,
        "itopk_sizes": itopk_sizes,
        "xb_shape": list(xb.shape),
        "xq_shape": list(xq.shape),
        "data_path": args.data,
        "timings_s": {"index_build": build_s},
        "itopk_results": [],
        "nprobe_results": [],  # alias for compare_cuvs_lib_json.py (nprobe := itopk)
        "cuvs": getattr(__import__("cuvs"), "__version__", "unknown"),
        "notes": "nprobe_results[].nprobe mirrors itopk_size for compare script reuse",
    }

    print(f"\n{backend} CAGRA results:")
    for itopk in itopk_sizes:
        try:
            search_params = cagra.SearchParams(
                itopk_size=itopk, search_width=args.search_width
            )
        except TypeError:
            search_params = cagra.SearchParams(itopk_size=itopk)

        for _ in range(max(0, args.warmup)):
            distances, neighbors = cagra.search(
                search_params, index, xq_g, args.k, **search_kw
            )
            _sync()

        t0 = time.perf_counter()
        distances, neighbors = cagra.search(
            search_params, index, xq_g, args.k, **search_kw
        )
        _sync()
        elapsed = time.perf_counter() - t0
        qps = xq.shape[0] / elapsed

        pred = device_ids_to_numpy(neighbors, cp)
        r = recall_at_k(pred, gt, args.k)

        lat_ms: list[float] = []
        p99_err = None
        sample_n = min(args.p99_sample, xq.shape[0])
        try:
            for i in range(sample_n):
                if xq_g.shape[0] >= 2:
                    q = xq_g[i : i + 2] if i + 2 <= xq_g.shape[0] else xq_g[i - 1 : i + 1]
                else:
                    q = xq_g[i : i + 1]
                s0 = time.perf_counter()
                cagra.search(search_params, index, q, args.k, **search_kw)
                _sync()
                lat_ms.append((time.perf_counter() - s0) * 1000.0)
        except Exception as exc:  # noqa: BLE001
            p99_err = str(exc)
            lat_ms = []
        p99 = float(np.percentile(lat_ms, 99)) if lat_ms else float("nan")
        if p99_err:
            print(f"  (p99 skipped: {p99_err.splitlines()[0][:120]})")

        print(
            f"itopk={itopk:4d} qps={qps:8.1f} p99_ms={p99:7.2f} recall@{args.k}={r:.4f}"
        )
        row = {
            "itopk_size": itopk,
            "nprobe": itopk,  # alias for compare_cuvs_lib_json.py
            "qps": qps,
            f"recall@{args.k}": r,
            "p99_ms": p99,
            "batch_search_s": elapsed,
            "p99_error": p99_err,
        }
        results["itopk_results"].append(row)
        results["nprobe_results"].append(row)

    if args.results_json:
        out = Path(args.results_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"results_json={out}")


if __name__ == "__main__":
    main()
