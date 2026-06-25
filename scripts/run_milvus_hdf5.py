import argparse
import time
import h5py
import numpy as np
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

DEFAULT_URI = "./milvus_sift.db"
DEFAULT_COLL = "sift1m_demo"
DEFAULT_DATA_PATH = "data/sift-128-euclidean.hdf5"
DEFAULT_K = 10
DEFAULT_NLIST = 1024
DEFAULT_NPROBES = "1,4,8,16,32"
DEFAULT_INSERT_BATCH = 50000
DEFAULT_P99_SAMPLE = 500


def recall_at_k(pred_ids, gt_ids, k):
    hits = 0
    nq = pred_ids.shape[0]
    for i in range(nq):
        hits += len(set(pred_ids[i, :k]).intersection(set(gt_ids[i, :k])))
    return hits / (nq * k)

parser = argparse.ArgumentParser(description="Milvus HDF5 ANN baseline")
parser.add_argument("--uri", default=DEFAULT_URI, help="Milvus URI or local Milvus Lite path")
parser.add_argument("--collection", default=DEFAULT_COLL, help="Collection name")
parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to sift-128-euclidean.hdf5")
parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k")
parser.add_argument("--nlist", type=int, default=DEFAULT_NLIST, help="IVF nlist")
parser.add_argument("--nprobes", default=DEFAULT_NPROBES, help="Comma-separated nprobe values")
parser.add_argument("--insert-batch", type=int, default=DEFAULT_INSERT_BATCH, help="Insert batch size")
parser.add_argument("--p99-sample", type=int, default=DEFAULT_P99_SAMPLE, help="Queries sampled for p99")
args = parser.parse_args()

nprobes = [int(x.strip()) for x in args.nprobes.split(",") if x.strip()]

print(f"milvus_uri={args.uri}")
print(f"collection={args.collection}, k={args.k}, nlist={args.nlist}, nprobes={nprobes}")

with h5py.File(args.data, "r") as f:
    xb = np.array(f["train"], dtype=np.float32)
    xq = np.array(f["test"], dtype=np.float32)
    gt = np.array(f["neighbors"], dtype=np.int64)

d = xb.shape[1]
print(f"xb={xb.shape}, xq={xq.shape}, gt={gt.shape}, dim={d}")

connections.connect(alias="default", uri=args.uri)
if utility.has_collection(args.collection):
    utility.drop_collection(args.collection)

schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=d),
    ],
    description="SIFT1M benchmark",
)
col = Collection(name=args.collection, schema=schema)

ids = np.arange(xb.shape[0], dtype=np.int64)
t0 = time.time()
for s in range(0, xb.shape[0], args.insert_batch):
    e = min(s + args.insert_batch, xb.shape[0])
    col.insert([ids[s:e].tolist(), xb[s:e].tolist()])
col.flush()
t1 = time.time()
print(f"insert_time_s={t1 - t0:.2f}")

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": args.nlist},
}
t0 = time.time()
col.create_index(field_name="vec", index_params=index_params)
t1 = time.time()
print(f"index_build_time_s={t1 - t0:.2f}")
print("index_params:", col.indexes[0].params)

col.load()
print("\nMilvus IVF_FLAT results:")
for nprobe in nprobes:
    search_params = {"metric_type": "L2", "params": {"nprobe": nprobe}}

    t0 = time.time()
    res = col.search(
        data=xq.tolist(),
        anns_field="vec",
        param=search_params,
        limit=args.k,
        output_fields=[],
    )
    t1 = time.time()
    qps = xq.shape[0] / (t1 - t0)

    pred = np.array([[hit.id for hit in row] for row in res], dtype=np.int64)
    r = recall_at_k(pred, gt, args.k)

    lat_ms = []
    for i in range(args.p99_sample):
        q = [xq[i].tolist()]
        s0 = time.time()
        _ = col.search(
            data=q,
            anns_field="vec",
            param=search_params,
            limit=args.k,
            output_fields=[],
        )
        s1 = time.time()
        lat_ms.append((s1 - s0) * 1000.0)
    p99 = float(np.percentile(lat_ms, 99))

    print(f"nprobe={nprobe:2d} qps={qps:8.1f} p99_ms={p99:7.2f} recall@{args.k}={r:.4f}")
