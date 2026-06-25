import argparse
import time
import hashlib
import h5py
import numpy as np
import faiss
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

DEFAULT_URI = "./milvus_sift.db"
DEFAULT_COLL = "sift1m_nprobe_diag"
DEFAULT_DATA_PATH = "data/sift-128-euclidean.hdf5"
DEFAULT_K = 10
DEFAULT_NLIST = 1024
DEFAULT_NPROBES = "1,2,4,8,16,32,64"
DEFAULT_INSERT_BATCH = 50000


def recall_at_k(pred_ids, gt_ids, k):
    hits = 0
    nq = pred_ids.shape[0]
    for i in range(nq):
        hits += len(set(pred_ids[i, :k]).intersection(set(gt_ids[i, :k])))
    return hits / (nq * k)


def digest_ids(pred_ids, prefix_queries=200):
    sample = pred_ids[:prefix_queries, :].astype(np.int64).tobytes()
    return hashlib.md5(sample).hexdigest()[:12]

parser = argparse.ArgumentParser(description="Diagnose Milvus nprobe effectiveness")
parser.add_argument("--uri", default=DEFAULT_URI, help="Milvus URI or local Milvus Lite path")
parser.add_argument("--collection", default=DEFAULT_COLL, help="Collection name")
parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to sift-128-euclidean.hdf5")
parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k")
parser.add_argument("--nlist", type=int, default=DEFAULT_NLIST, help="IVF nlist")
parser.add_argument("--nprobes", default=DEFAULT_NPROBES, help="Comma-separated nprobe values")
parser.add_argument("--insert-batch", type=int, default=DEFAULT_INSERT_BATCH, help="Insert batch size")
args = parser.parse_args()
nprobes = [int(x.strip()) for x in args.nprobes.split(",") if x.strip()]

print(f"milvus_uri={args.uri}")
print(f"collection={args.collection}, k={args.k}, nlist={args.nlist}, nprobes={nprobes}")

print("Loading dataset...")
with h5py.File(args.data, "r") as f:
    xb = np.array(f["train"], dtype=np.float32)
    xq = np.array(f["test"], dtype=np.float32)
    gt = np.array(f["neighbors"], dtype=np.int64)

d = xb.shape[1]
print(f"xb={xb.shape}, xq={xq.shape}, gt={gt.shape}, dim={d}")

print("\nFAISS reference curve (same nlist):")
ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, args.nlist, faiss.METRIC_L2)
ivf.train(xb[:200000])
ivf.add(xb)
for npb in nprobes:
    ivf.nprobe = npb
    t0 = time.time()
    _, I = ivf.search(xq, args.k)
    t1 = time.time()
    r = recall_at_k(I, gt, args.k)
    print(f"  faiss nprobe={npb:2d} qps={xq.shape[0]/(t1-t0):8.1f} recall@{args.k}={r:.4f}")

print("\nPreparing Milvus collection...")
connections.connect(alias="default", uri=args.uri)
if utility.has_collection(args.collection):
    utility.drop_collection(args.collection)

schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=d),
    ],
    description="nprobe diagnostics",
)
col = Collection(name=args.collection, schema=schema)

ids = np.arange(xb.shape[0], dtype=np.int64)
t0 = time.time()
for s in range(0, xb.shape[0], args.insert_batch):
    e = min(s + args.insert_batch, xb.shape[0])
    col.insert([ids[s:e].tolist(), xb[s:e].tolist()])
col.flush()
t1 = time.time()
print(f"Inserted in {t1-t0:.2f}s")

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": args.nlist},
}
col.create_index(field_name="vec", index_params=index_params)
print("index params from server:", col.indexes[0].params)
col.load()

print("\nMilvus run A: param={'metric_type':'L2','params':{'nprobe':N}}")
digests_a = []
for npb in nprobes:
    sp = {"metric_type": "L2", "params": {"nprobe": npb}}
    t0 = time.time()
    res = col.search(data=xq.tolist(), anns_field="vec", param=sp, limit=args.k, output_fields=[])
    t1 = time.time()
    pred = np.array([[hit.id for hit in row] for row in res], dtype=np.int64)
    r = recall_at_k(pred, gt, args.k)
    dg = digest_ids(pred)
    digests_a.append(dg)
    print(f"  milvusA nprobe={npb:2d} qps={xq.shape[0]/(t1-t0):8.1f} recall@{args.k}={r:.4f} digest={dg}")

print("\nMilvus run B: param={'nprobe':N} (legacy form)")
digests_b = []
for npb in nprobes:
    sp = {"nprobe": npb}
    t0 = time.time()
    res = col.search(data=xq.tolist(), anns_field="vec", param=sp, limit=args.k, output_fields=[])
    t1 = time.time()
    pred = np.array([[hit.id for hit in row] for row in res], dtype=np.int64)
    r = recall_at_k(pred, gt, args.k)
    dg = digest_ids(pred)
    digests_b.append(dg)
    print(f"  milvusB nprobe={npb:2d} qps={xq.shape[0]/(t1-t0):8.1f} recall@{args.k}={r:.4f} digest={dg}")

print("\nDigest variability check:")
print("  run A unique digests:", len(set(digests_a)), set(digests_a))
print("  run B unique digests:", len(set(digests_b)), set(digests_b))
if len(set(digests_a)) == 1 and len(set(digests_b)) == 1:
    print("  WARNING: nprobe appears ineffective (results identical across sweep).")
else:
    print("  OK: nprobe changes returned neighbors for at least one parameter form.")
