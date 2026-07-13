import argparse
import time
import h5py
import numpy as np
from pymilvus import MilvusClient, DataType

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


parser = argparse.ArgumentParser(description="Milvus HDF5 ANN baseline / Layer-3 GPU smoke")
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
    choices=["IVF_FLAT", "GPU_IVF_FLAT"],
    help="Index type (GPU_IVF_FLAT requires HIP Milvus Layer 3)",
)
parser.add_argument(
    "--cache-dataset-on-device",
    action="store_true",
    help="GPU index param: keep dataset on device (GPU_IVF_FLAT)",
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
args = parser.parse_args()

nprobes = [int(x.strip()) for x in args.nprobes.split(",") if x.strip()]

print(f"milvus_uri={args.uri}")
print(
    f"collection={args.collection}, k={args.k}, nlist={args.nlist}, "
    f"nprobes={nprobes}, index_type={args.index_type}"
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
print(f"insert_time_s={t1 - t0:.2f}")

index_params = client.prepare_index_params()
idx_extra = {"nlist": args.nlist}
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
print(f"index_build_time_s={t1 - t0:.2f}")
print(
    "index_params:",
    {"index_type": args.index_type, "metric_type": "L2", "params": idx_extra},
)

client.load_collection(collection_name=args.collection)
print(f"\nMilvus {args.index_type} results:")
for nprobe in nprobes:
    search_params = {"metric_type": "L2", "params": {"nprobe": nprobe}}

    t0 = time.time()
    res = client.search(
        collection_name=args.collection,
        data=xq.tolist(),
        anns_field="vec",
        search_params=search_params,
        limit=args.k,
        output_fields=[],
    )
    t1 = time.time()
    qps = xq.shape[0] / (t1 - t0)

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

    print(f"nprobe={nprobe:2d} qps={qps:8.1f} p99_ms={p99:7.2f} recall@{args.k}={r:.4f}")
