import time
import hashlib
import h5py
import numpy as np
import faiss
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

MILVUS_URI = "./milvus_sift.db"
COLL = "sift1m_nprobe_diag"
DATA_PATH = "data/sift-128-euclidean.hdf5"
K = 10
NLIST = 1024
NPROBES = [1, 2, 4, 8, 16, 32, 64]
INSERT_BATCH = 50000


def recall_at_k(pred_ids, gt_ids, k):
    hits = 0
    nq = pred_ids.shape[0]
    for i in range(nq):
        hits += len(set(pred_ids[i, :k]).intersection(set(gt_ids[i, :k])))
    return hits / (nq * k)


def digest_ids(pred_ids, prefix_queries=200):
    sample = pred_ids[:prefix_queries, :].astype(np.int64).tobytes()
    return hashlib.md5(sample).hexdigest()[:12]


print("Loading dataset...")
with h5py.File(DATA_PATH, "r") as f:
    xb = np.array(f["train"], dtype=np.float32)
    xq = np.array(f["test"], dtype=np.float32)
    gt = np.array(f["neighbors"], dtype=np.int64)

d = xb.shape[1]
print(f"xb={xb.shape}, xq={xq.shape}, gt={gt.shape}, dim={d}")

print("\nFAISS reference curve (same nlist):")
ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, NLIST, faiss.METRIC_L2)
ivf.train(xb[:200000])
ivf.add(xb)
for npb in NPROBES:
    ivf.nprobe = npb
    t0 = time.time()
    _, I = ivf.search(xq, K)
    t1 = time.time()
    r = recall_at_k(I, gt, K)
    print(f"  faiss nprobe={npb:2d} qps={xq.shape[0]/(t1-t0):8.1f} recall@{K}={r:.4f}")

print("\nPreparing Milvus collection...")
connections.connect(alias="default", uri=MILVUS_URI)
if utility.has_collection(COLL):
    utility.drop_collection(COLL)

schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=d),
    ],
    description="nprobe diagnostics",
)
col = Collection(name=COLL, schema=schema)

ids = np.arange(xb.shape[0], dtype=np.int64)
t0 = time.time()
for s in range(0, xb.shape[0], INSERT_BATCH):
    e = min(s + INSERT_BATCH, xb.shape[0])
    col.insert([ids[s:e].tolist(), xb[s:e].tolist()])
col.flush()
t1 = time.time()
print(f"Inserted in {t1-t0:.2f}s")

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": NLIST},
}
col.create_index(field_name="vec", index_params=index_params)
print("index params from server:", col.indexes[0].params)
col.load()

print("\nMilvus run A: param={'metric_type':'L2','params':{'nprobe':N}}")
digests_a = []
for npb in NPROBES:
    sp = {"metric_type": "L2", "params": {"nprobe": npb}}
    t0 = time.time()
    res = col.search(data=xq.tolist(), anns_field="vec", param=sp, limit=K, output_fields=[])
    t1 = time.time()
    pred = np.array([[hit.id for hit in row] for row in res], dtype=np.int64)
    r = recall_at_k(pred, gt, K)
    dg = digest_ids(pred)
    digests_a.append(dg)
    print(f"  milvusA nprobe={npb:2d} qps={xq.shape[0]/(t1-t0):8.1f} recall@{K}={r:.4f} digest={dg}")

print("\nMilvus run B: param={'nprobe':N} (legacy form)")
digests_b = []
for npb in NPROBES:
    sp = {"nprobe": npb}
    t0 = time.time()
    res = col.search(data=xq.tolist(), anns_field="vec", param=sp, limit=K, output_fields=[])
    t1 = time.time()
    pred = np.array([[hit.id for hit in row] for row in res], dtype=np.int64)
    r = recall_at_k(pred, gt, K)
    dg = digest_ids(pred)
    digests_b.append(dg)
    print(f"  milvusB nprobe={npb:2d} qps={xq.shape[0]/(t1-t0):8.1f} recall@{K}={r:.4f} digest={dg}")

print("\nDigest variability check:")
print("  run A unique digests:", len(set(digests_a)), set(digests_a))
print("  run B unique digests:", len(set(digests_b)), set(digests_b))
if len(set(digests_a)) == 1 and len(set(digests_b)) == 1:
    print("  WARNING: nprobe appears ineffective (results identical across sweep).")
else:
    print("  OK: nprobe changes returned neighbors for at least one parameter form.")
