import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from pymilvus import DataType, MilvusClient

DEFAULT_URI = "./milvus_sift.db"
DEFAULT_COLL = "sift1m_demo"
DEFAULT_DATA_PATH = "data/sift-128-euclidean.hdf5"
DEFAULT_K = 10
DEFAULT_NLIST = 1024
DEFAULT_NPROBES = "1,4,8,16,32"
DEFAULT_INSERT_BATCH = 50000
DEFAULT_P99_SAMPLE = 500
DEFAULT_INDEX_TYPE = "IVF_FLAT"


def recall_at_k(pred_ids, gt_ids, k):
    hits = 0
    nq = pred_ids.shape[0]
    for i in range(nq):
        hits += len(set(pred_ids[i, :k]).intersection(set(gt_ids[i, :k])))
    return hits / (nq * k)


def to_pred_ids(search_res):
    def hit_id(hit):
        if isinstance(hit, dict):
            return int(hit["id"])
        return int(hit.id)

    return np.array([[hit_id(hit) for hit in row] for row in search_res], dtype=np.int64)


def wait_ready_for_search(client: MilvusClient, collection: str, timeout_s: float) -> dict:
    """Wait until an index exists and the collection can be loaded (sealed path)."""
    deadline = time.time() + max(0.0, timeout_s)
    last_err = None
    while time.time() < deadline:
        try:
            idxs = client.list_indexes(collection_name=collection)
            if not idxs:
                last_err = RuntimeError("list_indexes empty")
                time.sleep(2.0)
                continue
            index_name = idxs[0] if isinstance(idxs[0], str) else str(idxs[0])
            desc = client.describe_index(collection_name=collection, index_name=index_name)
            client.load_collection(collection_name=collection)
            state = None
            if hasattr(client, "get_load_state"):
                state = client.get_load_state(collection_name=collection)
            print(f"load_ready index={index_name!r} desc={desc} state={state}")
            return {
                "index_name": index_name,
                "index_desc": desc if isinstance(desc, dict) else str(desc),
                "load_state": str(state),
            }
        except Exception as exc:  # noqa: BLE001 - poll until timeout
            last_err = exc
            time.sleep(2.0)
    raise TimeoutError(
        f"collection {collection!r} not ready for search within {timeout_s}s: {last_err}"
    )


parser = argparse.ArgumentParser(
    description="Milvus HDF5 ANN baseline / Layer-3 smoke / Layer-4 GPU benchmark"
)
parser.add_argument("--uri", default=DEFAULT_URI, help="Milvus URI or local Milvus Lite path")
parser.add_argument("--collection", default=DEFAULT_COLL, help="Collection name")
parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to sift-128-euclidean.hdf5")
parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k")
parser.add_argument("--nlist", type=int, default=DEFAULT_NLIST, help="IVF nlist")
parser.add_argument("--nprobes", default=DEFAULT_NPROBES, help="Comma-separated nprobe values")
parser.add_argument("--insert-batch", type=int, default=DEFAULT_INSERT_BATCH, help="Insert batch size")
parser.add_argument("--p99-sample", type=int, default=DEFAULT_P99_SAMPLE, help="Queries sampled for p99")
parser.add_argument(
    "--index-type",
    default=DEFAULT_INDEX_TYPE,
    choices=["IVF_FLAT", "GPU_IVF_FLAT", "IVF_PQ", "GPU_IVF_PQ", "GPU_CAGRA"],
    help="Index type (GPU_* requires HIP/CUDA Milvus GPU build)",
)
parser.add_argument(
    "--m",
    type=int,
    default=16,
    help="PQ sub-quantizers (IVF_PQ / GPU_IVF_PQ); must divide vector dim "
    "(SIFT-128: 8/16/32/64; GIST-960: e.g. 32)",
)
parser.add_argument(
    "--nbits",
    type=int,
    default=8,
    help="PQ bits per sub-quantizer (IVF_PQ / GPU_IVF_PQ); typically 8",
)
parser.add_argument(
    "--graph-degree",
    type=int,
    default=32,
    help="GPU_CAGRA index param graph_degree",
)
parser.add_argument(
    "--intermediate-graph-degree",
    type=int,
    default=64,
    help="GPU_CAGRA index param intermediate_graph_degree",
)
parser.add_argument(
    "--itopk-sizes",
    default="",
    help="GPU_CAGRA search sweep (comma-separated itopk_size). "
    "If empty and index is GPU_CAGRA, defaults to 32,64,128,256. "
    "IVF indexes ignore this and use --nprobes.",
)
parser.add_argument(
    "--search-width",
    type=int,
    default=1,
    help="GPU_CAGRA search_params search_width",
)
parser.add_argument(
    "--build-algo",
    default="NN_DESCENT",
    help="GPU_CAGRA index param build_algo (Milvus/Knowhere: IVF_PQ | NN_DESCENT). "
    "Default NN_DESCENT — required on gfx1100 where hipVS IVF_PQ graph build fails.",
)
parser.add_argument(
    "--cache-dataset-on-device",
    action="store_true",
    help="GPU index param: keep dataset on device (GPU_* indexes)",
)
parser.add_argument(
    "--max-train-rows",
    type=int,
    default=0,
    help="If >0, use only the first N train vectors (Layer-3 smoke)",
)
parser.add_argument(
    "--max-query-rows",
    type=int,
    default=0,
    help="If >0, use only the first N query vectors (Layer-3 smoke)",
)
parser.add_argument(
    "--flush",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Flush after insert to seal segments (default: on for GPU_* indexes). "
    "Without flush, QueryNode may search growing CPU IVF_FLAT_CC instead of sealed GPU.",
)
parser.add_argument(
    "--index-wait-s",
    type=float,
    default=180.0,
    help="Max seconds to wait after flush/create_index before load/search",
)
parser.add_argument(
    "--results-json",
    default="",
    help="If set, write timings + per-nprobe metrics to this JSON path",
)
parser.add_argument(
    "--search-warmup",
    type=int,
    default=0,
    help="Full-batch search warm-up runs per sweep point (discarded; default 0)",
)
parser.add_argument(
    "--search-runs",
    type=int,
    default=1,
    help="Timed full-batch search runs per sweep point; report median QPS (default 1)",
)
args = parser.parse_args()
if args.search_warmup < 0:
    raise SystemExit("--search-warmup must be >= 0")
if args.search_runs < 1:
    raise SystemExit("--search-runs must be >= 1")

is_cagra = args.index_type == "GPU_CAGRA"
nprobes = [int(x.strip()) for x in args.nprobes.split(",") if x.strip()]
if is_cagra:
    itopk_raw = args.itopk_sizes.strip() or "32,64,128,256"
    sweep = [int(x.strip()) for x in itopk_raw.split(",") if x.strip()]
else:
    sweep = nprobes
do_flush = args.flush if args.flush is not None else args.index_type.startswith("GPU_")

print(f"milvus_uri={args.uri}")
print(
    f"collection={args.collection}, k={args.k}, nlist={args.nlist}, "
    f"sweep={sweep}, index_type={args.index_type}, flush={do_flush}"
)
if "PQ" in args.index_type:
    print(f"pq_params: m={args.m}, nbits={args.nbits}")
if is_cagra:
    print(
        f"cagra_params: graph_degree={args.graph_degree}, "
        f"intermediate_graph_degree={args.intermediate_graph_degree}, "
        f"build_algo={args.build_algo}, "
        f"search_width={args.search_width}, itopk_sizes={sweep}"
    )

with h5py.File(args.data, "r") as f:
    xb = np.array(f["train"], dtype=np.float32)
    xq = np.array(f["test"], dtype=np.float32)
    gt = np.array(f["neighbors"], dtype=np.int64)

if args.max_train_rows and args.max_train_rows > 0:
    xb = xb[: args.max_train_rows]
if args.max_query_rows and args.max_query_rows > 0:
    xq = xq[: args.max_query_rows]
    gt = gt[: args.max_query_rows]

d = xb.shape[1]
print(f"xb={xb.shape}, xq={xq.shape}, gt={gt.shape}, dim={d}")
if "PQ" in args.index_type and d % args.m != 0:
    raise SystemExit(
        f"PQ m={args.m} must divide dim={d} "
        f"(SIFT-128: 8/16/32/64; GIST-960: 32/60/64/80/...)"
    )

results = {
    "uri": args.uri,
    "collection": args.collection,
    "index_type": args.index_type,
    "flush": do_flush,
    "nlist": None if is_cagra else args.nlist,
    "m": args.m if "PQ" in args.index_type else None,
    "nbits": args.nbits if "PQ" in args.index_type else None,
    "graph_degree": args.graph_degree if is_cagra else None,
    "intermediate_graph_degree": args.intermediate_graph_degree if is_cagra else None,
    "build_algo": args.build_algo if is_cagra else None,
    "search_width": args.search_width if is_cagra else None,
    "k": args.k,
    "nprobes": nprobes if not is_cagra else None,
    "itopk_sizes": sweep if is_cagra else None,
    "xb_shape": list(xb.shape),
    "xq_shape": list(xq.shape),
    "data_path": args.data,
    "search_warmup": args.search_warmup,
    "search_runs": args.search_runs,
    "timings_s": {},
    "load": None,
    "nprobe_results": [],
    "itopk_results": [],
}

client = MilvusClient(uri=args.uri)
if client.has_collection(args.collection):
    client.drop_collection(args.collection)

schema = client.create_schema(auto_id=False, enable_dynamic_fields=False)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=d)
client.create_collection(collection_name=args.collection, schema=schema)

ids = np.arange(xb.shape[0], dtype=np.int64)
t0 = time.time()
for s in range(0, xb.shape[0], args.insert_batch):
    e = min(s + args.insert_batch, xb.shape[0])
    batch = [{"id": int(i), "vec": v.tolist()} for i, v in zip(ids[s:e], xb[s:e])]
    client.insert(collection_name=args.collection, data=batch)
t1 = time.time()
results["timings_s"]["insert"] = t1 - t0
print(f"insert_time_s={t1 - t0:.2f}")

if do_flush:
    t0 = time.time()
    client.flush(args.collection)
    t1 = time.time()
    results["timings_s"]["flush"] = t1 - t0
    print(f"flush_time_s={t1 - t0:.2f}")

index_params = client.prepare_index_params()
if is_cagra:
    idx_extra = {
        "graph_degree": args.graph_degree,
        "intermediate_graph_degree": args.intermediate_graph_degree,
        "build_algo": args.build_algo,
    }
else:
    idx_extra = {"nlist": args.nlist}
    if "PQ" in args.index_type:
        idx_extra["m"] = args.m
        idx_extra["nbits"] = args.nbits
if args.index_type.startswith("GPU_") and args.cache_dataset_on_device:
    idx_extra["cache_dataset_on_device"] = True
index_params.add_index(
    field_name="vec",
    index_type=args.index_type,
    metric_type="L2",
    params=idx_extra,
)
t0 = time.time()
client.create_index(collection_name=args.collection, index_params=index_params)
t1 = time.time()
results["timings_s"]["index_build"] = t1 - t0
print(f"index_build_time_s={t1 - t0:.2f}")
print(
    "index_params:",
    {"index_type": args.index_type, "metric_type": "L2", "params": idx_extra},
)

if do_flush:
    t0 = time.time()
    client.flush(args.collection)
    results["timings_s"]["post_index_flush"] = time.time() - t0
    print(f"post_index_flush_time_s={results['timings_s']['post_index_flush']:.2f}")
    print(f"waiting up to {args.index_wait_s:.0f}s for sealed index / load...")
    t0 = time.time()
    results["load"] = wait_ready_for_search(client, args.collection, args.index_wait_s)
    results["timings_s"]["index_wait_load"] = time.time() - t0
else:
    t0 = time.time()
    client.load_collection(collection_name=args.collection)
    results["timings_s"]["load"] = time.time() - t0
    results["load"] = {"load_state": "Loaded"}

print(f"\nMilvus {args.index_type} results:")
if args.search_warmup or args.search_runs != 1:
    print(
        f"search_protocol: warmup={args.search_warmup} "
        f"timed_runs={args.search_runs} (report median QPS)"
    )
xq_list = xq.tolist()
for sweep_val in sweep:
    if is_cagra:
        search_params = {
            "metric_type": "L2",
            "params": {
                "itopk_size": sweep_val,
                "search_width": args.search_width,
            },
        }
        label = f"itopk={sweep_val:4d}"
    else:
        search_params = {"metric_type": "L2", "params": {"nprobe": sweep_val}}
        label = f"nprobe={sweep_val:2d}"

    for _ in range(args.search_warmup):
        _ = client.search(
            collection_name=args.collection,
            data=xq_list,
            anns_field="vec",
            search_params=search_params,
            limit=args.k,
            output_fields=[],
        )

    qps_runs = []
    res = None
    for _ in range(args.search_runs):
        t0 = time.time()
        res = client.search(
            collection_name=args.collection,
            data=xq_list,
            anns_field="vec",
            search_params=search_params,
            limit=args.k,
            output_fields=[],
        )
        t1 = time.time()
        qps_runs.append(xq.shape[0] / (t1 - t0))
    qps = float(np.median(qps_runs))

    pred = to_pred_ids(res)
    r = recall_at_k(pred, gt, args.k)

    lat_ms = []
    sample_n = min(args.p99_sample, xq.shape[0])
    for i in range(sample_n):
        q = [xq[i].tolist()]
        s0 = time.time()
        _ = client.search(
            collection_name=args.collection,
            data=q,
            anns_field="vec",
            search_params=search_params,
            limit=args.k,
            output_fields=[],
        )
        s1 = time.time()
        lat_ms.append((s1 - s0) * 1000.0)
    p99 = float(np.percentile(lat_ms, 99)) if lat_ms else float("nan")

    if args.search_runs > 1:
        print(
            f"{label} qps_median={qps:8.1f} "
            f"qps_runs={[round(x, 1) for x in qps_runs]} "
            f"p99_ms={p99:7.2f} recall@{args.k}={r:.4f}"
        )
    else:
        print(f"{label} qps={qps:8.1f} p99_ms={p99:7.2f} recall@{args.k}={r:.4f}")
    row = {
        "qps": qps,
        "qps_runs": qps_runs,
        "qps_median": qps,
        "search_warmup": args.search_warmup,
        "search_runs": args.search_runs,
        f"recall@{args.k}": r,
        "p99_ms": p99,
    }
    if is_cagra:
        row["itopk_size"] = sweep_val
        row["nprobe"] = sweep_val  # alias for compare tooling
        results["itopk_results"].append(row)
        results["nprobe_results"].append(row)
    else:
        row["nprobe"] = sweep_val
        results["nprobe_results"].append(row)

if args.results_json:
    out = Path(args.results_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"results_json={out}")
