#!/usr/bin/env python3
"""Minimal cuVS/hipVS CAGRA filtered-search repro (no Knowhere).

Mirrors Catch2 bitset gaps seen on gfx1100 after NN_DESCENT:
  - unfiltered self/query search should be high recall
  - ~40% filter (Knowhere "Search With Bitset") 
  - dense 64-row Simple Bitset pattern from test_gpu_search.cc

Polarity:
  Knowhere BitsetView: bit SET  => excluded from results
  cuVS filters.from_bitset: bit 0 => excluded, bit 1 => allowed
  This script converts Knowhere bytes -> cuVS uint32 words.

Usage (AMD):
  source ~/hipvs-bench-venv/bin/activate
  bash scripts/run_hipvs_cagra_filter_repro.sh

Usage (direct):
  python3 scripts/probe_cuvs_cagra_filter.py --build-algo nn_descent
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

# Catch2 Simple Bitset pattern (test_gpu_search.cc) — Knowhere polarity
SIMPLE_BITSET_BYTES = bytes(
    [
        0b10100010,
        0b00100011,
        0b10100010,
        0b00100111,
        0b10100000,
        0b00000000,
        0b00000010,
        0b11100011,
    ]
)


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    hits = 0
    nq = pred.shape[0]
    for i in range(nq):
        hits += len(set(pred[i, :k]).intersection(set(gt[i, :k])))
    return hits / (nq * k)


def exact_neighbors_filtered(
    xb: np.ndarray, xq: np.ndarray, k: int, allowed: np.ndarray
) -> np.ndarray:
    """Brute-force top-k under sqeuclidean, only among allowed[i]==True."""
    nq = xq.shape[0]
    allow_idx = np.flatnonzero(allowed)
    if allow_idx.size == 0:
        raise SystemExit("no allowed vectors for GT")
    k_eff = min(k, allow_idx.size)
    sub = xb[allow_idx].astype(np.float64)
    xb_norm = np.sum(sub**2, axis=1)
    out = np.full((nq, k), -1, dtype=np.int64)
    for s in range(0, nq, 64):
        e = min(s + 64, nq)
        q = xq[s:e].astype(np.float64)
        dots = q @ sub.T
        q_norm = np.sum(q**2, axis=1, keepdims=True)
        dist = xb_norm[None, :] + q_norm - 2.0 * dots
        part = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        row = np.arange(e - s)[:, None]
        order = np.argsort(dist[row, part], axis=1)
        local = part[row, order]
        out[s:e, :k_eff] = allow_idx[local]
    return out


def knowhere_bytes_to_allowed(data: bytes, n: int) -> np.ndarray:
    """Knowhere: bit set => excluded. Returns bool allowed[n]."""
    allowed = np.ones(n, dtype=bool)
    for i in range(n):
        if data[i // 8] & (1 << (i % 8)):
            allowed[i] = False
    return allowed


def allowed_to_cuvs_bitset_u32(allowed: np.ndarray) -> np.ndarray:
    """Pack allowed mask into cuVS uint32 words (bit1=allowed)."""
    n = int(allowed.shape[0])
    n_words = (n + 31) // 32
    out = np.zeros(n_words, dtype=np.uint32)
    for i in range(n):
        if allowed[i]:
            out[i // 32] |= np.uint32(1) << (i % 32)
    return out


def random_allowed(n: int, exclude_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    allowed = np.ones(n, dtype=bool)
    n_ex = int(round(n * exclude_frac))
    if n_ex > 0:
        allowed[rng.choice(n, size=n_ex, replace=False)] = False
    return allowed


def device_ids_to_numpy(neighbors, cp_mod) -> np.ndarray:
    if hasattr(neighbors, "__cuda_array_interface__"):
        raw = np.asarray(cp_mod.asarray(neighbors).get())
    elif hasattr(neighbors, "copy_to_host"):
        raw = np.asarray(neighbors.copy_to_host())
    else:
        raw = np.asarray(neighbors)
    # hipVS/cuVS often use uint32 with 0xFFFFFFFF as "no neighbor"
    if raw.dtype == np.uint32 or raw.dtype == np.uint64:
        out = raw.astype(np.int64, copy=True)
        out[raw == np.iinfo(raw.dtype).max] = -1
        return out
    return raw.astype(np.int64, copy=False)


def make_filter(cp, filters_mod, allowed: np.ndarray):
    words = allowed_to_cuvs_bitset_u32(allowed)
    bitset_g = cp.asarray(words)
    # Prefer from_bitset; try a few historical names.
    for name in ("from_bitset", "bitset", "BitsetFilter"):
        fn = getattr(filters_mod, name, None)
        if fn is None:
            continue
        try:
            return fn(bitset_g), f"filters.{name}"
        except TypeError:
            try:
                return fn(bitset_g.view()), f"filters.{name}(view)"
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"filters.{name} failed: {exc}")
    raise SystemExit(
        "Could not construct cuVS/hipVS bitset filter. "
        f"filters attrs={dir(filters_mod)}"
    )


def cagra_search(cagra, search_params, index, queries, k, filt=None, resources=None):
    kw = {}
    if resources is not None:
        kw["resources"] = resources
    if filt is not None:
        # cagra.search uses filter=; some builds used prefilter=
        for key in ("filter", "prefilter"):
            try:
                return cagra.search(
                    search_params, index, queries, k, **{key: filt}, **kw
                )
            except TypeError as exc:
                last = exc
                continue
        raise SystemExit(f"cagra.search rejected filter/prefilter: {last}")
    return cagra.search(search_params, index, queries, k, **kw)


def summarize_case(
    tag: str,
    pred: np.ndarray,
    gt: np.ndarray,
    k: int,
    allowed: np.ndarray,
) -> dict:
    neg = int(np.sum(pred < 0))
    # how many hits land on excluded ids (filter leak)
    leak = 0
    oob = 0
    total = pred.size
    n_allow = int(allowed.shape[0])
    for v in pred.ravel():
        vi = int(v)
        if vi < 0:
            continue
        if vi >= n_allow:
            oob += 1
            continue
        if not allowed[vi]:
            leak += 1
    # ignore invalid ids in recall (treat as miss)
    pred_safe = pred.copy()
    pred_safe[(pred_safe < 0) | (pred_safe >= n_allow)] = -2
    r = recall_at_k(pred_safe, gt, k)
    sample = []
    for i in range(min(6, pred.shape[0])):
        sample.append(
            {
                "q": i,
                "pred": pred[i, :k].tolist(),
                "gt": gt[i, :k].tolist(),
            }
        )
    row = {
        "tag": tag,
        "nq": int(pred.shape[0]),
        "k": k,
        "recall": float(r),
        "neg1": neg,
        "oob_ids": oob,
        "filter_leaks": leak,
        "allowed_frac": float(allowed.mean()),
        "sample": sample,
    }
    print(
        f"[{tag}] recall@{k}={r:.4f} neg1={neg}/{total} "
        f"oob={oob} leaks={leak} allowed_frac={allowed.mean():.3f}"
    )
    for s in sample[:4]:
        print(f"  q{s['q']}: pred={s['pred']} gt={s['gt']}")
    return row


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
            if "Card series" in line or "Card model" in line:
                return "hipvs", line.split(":", 1)[-1].strip()
        return "hipvs", "AMD GPU"
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return "unknown", "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-query", type=int, default=200)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--graph-degree", type=int, default=32)
    ap.add_argument("--intermediate-graph-degree", type=int, default=64)
    ap.add_argument("--itopk-size", type=int, default=128)
    ap.add_argument("--build-algo", default="nn_descent")
    ap.add_argument("--exclude-frac", type=float, default=0.4)
    ap.add_argument("--results-json", default="")
    args = ap.parse_args()

    import cupy as cp
    from cuvs.neighbors import cagra

    try:
        from cuvs.neighbors import filters
    except ImportError:
        try:
            from cuvs.neighbors import filtering as filters  # type: ignore
        except ImportError as exc:
            print(f"ERROR: no cuvs.neighbors.filters ({exc})", file=sys.stderr)
            return 2

    backend, gpu = detect_backend()
    print(f"host={platform.node()} backend={backend} gpu={gpu!r}")
    print(
        f"n_train={args.n_train} n_query={args.n_query} dim={args.dim} "
        f"k={args.k} build_algo={args.build_algo} itopk={args.itopk_size}"
    )
    print(f"filters module={getattr(filters, '__name__', filters)} attrs_sample="
          f"{[a for a in dir(filters) if not a.startswith('_')][:20]}")

    rng = np.random.default_rng(args.seed)
    # Match Catch2 GenDataSet spirit: shared seed stream; queries = first n_query of train
    xb = rng.standard_normal((args.n_train, args.dim), dtype=np.float32)
    xq = xb[: args.n_query].copy()

    resources = None
    try:
        from cuvs.common import Resources

        resources = Resources()
    except Exception:  # noqa: BLE001
        pass

    algo = args.build_algo.strip().lower().replace("-", "_")
    try:
        build_params = cagra.IndexParams(
            metric="sqeuclidean",
            graph_degree=args.graph_degree,
            intermediate_graph_degree=args.intermediate_graph_degree,
            build_algo=algo,
        )
    except TypeError:
        build_params = cagra.IndexParams(
            metric="sqeuclidean",
            graph_degree=args.graph_degree,
            build_algo=algo,
        )
    if hasattr(build_params, "build_algo"):
        print(f"IndexParams.build_algo={build_params.build_algo!r}")

    xb_g = cp.asarray(xb)
    xq_g = cp.asarray(xq)
    build_kw = {"resources": resources} if resources is not None else {}
    print("building CAGRA…")
    index = cagra.build(build_params, xb_g, **build_kw)
    if resources is not None and hasattr(resources, "sync"):
        resources.sync()
    else:
        cp.cuda.Device().synchronize()

    try:
        search_params = cagra.SearchParams(itopk_size=args.itopk_size)
    except TypeError:
        search_params = cagra.SearchParams()

    cases: list[dict] = []

    # 1) Unfiltered
    dist, neigh = cagra_search(
        cagra, search_params, index, xq_g, args.k, filt=None, resources=resources
    )
    if resources is not None and hasattr(resources, "sync"):
        resources.sync()
    pred = device_ids_to_numpy(neigh, cp)
    allowed_all = np.ones(args.n_train, dtype=bool)
    gt = exact_neighbors_filtered(xb, xq, args.k, allowed_all)
    cases.append(summarize_case("unfiltered", pred, gt, args.k, allowed_all))

    # 2) ~40% excluded (Catch2 bitset percentage 0.4)
    allowed40 = random_allowed(args.n_train, args.exclude_frac, seed=args.seed + 7)
    filt40, how = make_filter(cp, filters, allowed40)
    print(f"filter ctor: {how}")
    dist, neigh = cagra_search(
        cagra, search_params, index, xq_g, args.k, filt=filt40, resources=resources
    )
    if resources is not None and hasattr(resources, "sync"):
        resources.sync()
    pred = device_ids_to_numpy(neigh, cp)
    gt = exact_neighbors_filtered(xb, xq, args.k, allowed40)
    cases.append(summarize_case("filter_40pct", pred, gt, args.k, allowed40))

    # 3) Simple bitset on 64-row index (Catch2 pattern)
    n64 = 64
    xb64 = xb[:n64].copy()
    xq64 = xb64  # self-search like Catch2
    allowed64 = knowhere_bytes_to_allowed(SIMPLE_BITSET_BYTES, n64)
    print(
        f"simple_bitset: allowed={int(allowed64.sum())}/{n64} "
        f"excluded={int((~allowed64).sum())}"
    )
    build_params64 = build_params
    try:
        build_params64 = cagra.IndexParams(
            metric="sqeuclidean",
            graph_degree=min(32, args.graph_degree),
            intermediate_graph_degree=min(32, args.intermediate_graph_degree),
            build_algo=algo,
        )
    except TypeError:
        pass
    xb64_g = cp.asarray(xb64)
    xq64_g = cp.asarray(xq64)
    index64 = cagra.build(build_params64, xb64_g, **build_kw)
    if resources is not None and hasattr(resources, "sync"):
        resources.sync()
    try:
        sp64 = cagra.SearchParams(itopk_size=32)
    except TypeError:
        sp64 = search_params
    filt64, how64 = make_filter(cp, filters, allowed64)
    print(f"simple_bitset filter ctor: {how64}")
    dist, neigh = cagra_search(
        cagra, sp64, index64, xq64_g, args.k, filt=filt64, resources=resources
    )
    if resources is not None and hasattr(resources, "sync"):
        resources.sync()
    pred = device_ids_to_numpy(neigh, cp)
    gt = exact_neighbors_filtered(xb64, xq64, args.k, allowed64)
    cases.append(summarize_case("simple_bitset_64", pred, gt, args.k, allowed64))

    # Ownership hint
    uf = cases[0]["recall"]
    f40 = cases[1]["recall"]
    sb = cases[2]["recall"]
    print("\n== OWNERSHIP ==")
    if uf >= 0.9 and (f40 < 0.3 or sb < 0.3 or cases[2]["neg1"] == cases[2]["nq"] * args.k):
        print(
            "OWNER: hipVS/cuVS CAGRA filtered search "
            "(unfiltered OK, filter recall dead / all -1)."
        )
        print("Knowhere wiring unlikely — Catch2 bitset maps onto this API.")
    elif uf >= 0.9 and f40 >= 0.7 and sb >= 0.8:
        print(
            "OWNER: Knowhere bitset wiring/polarity "
            "(pure hipVS filter path looks healthy)."
        )
    else:
        print(
            f"OWNER: unclear (unfiltered={uf:.3f} filter40={f40:.3f} "
            f"simple={sb:.3f}) — inspect samples / filter ctor."
        )

    out = {
        "protocol": "library_cuvs_cagra_filter_repro",
        "backend": backend,
        "gpu_name": gpu,
        "host": platform.node(),
        "build_algo": args.build_algo,
        "itopk_size": args.itopk_size,
        "cases": cases,
        "cuvs": getattr(__import__("cuvs"), "__version__", "unknown"),
    }
    if args.results_json:
        path = Path(args.results_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"results_json={path}")
    else:
        print(json.dumps({"cases": [
            {k: c[k] for k in ("tag", "recall", "neg1", "filter_leaks", "allowed_frac")}
            for c in cases
        ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
