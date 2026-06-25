import time
import h5py
import numpy as np
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

MILVUS_URI = "./milvus_sift.db"
COLL = "sift1m_demo"
DATA_PATH = "data/sift-128-euclidean.hdf5"
K = 10
NLIST = 1024
NPROBES = [1, 4, 8, 16, 32]
INSERT_BATCH = 50000
P99_SAMPLE = 500


def recall_at_k(pred_ids, gt_ids, k):
    hits = 0
    nq = pred_ids.shape[0]
    for i in range(nq):
        hits += len(set(pred_ids[i, :k]).intersection(set(gt_ids[i, :k])))
    return hits / (nq * k)


with h5py.File(DATA_PATH, "r") as f:
    xb = np.array(f["train"], dtype=np.float32)
    xq = np.array(f["test"], dtype=np.float32)
    gt = np.array(f["neighbors"], dtype=np.int64)

d = xb.shape[1]
print(f"xb={xb.shape}, xq={xq.shape}, gt={gt.shape}, dim={d}")

connections.connect(alias="default", uri=MILVUS_URI)
if utility.has_collection(COLL):
    utility.drop_collection(COLL)

schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=d),
    ],
    description="SIFT1M benchmark",
)
col = Collection(name=COLL, schema=schema)

ids = np.arange(xb.shape[0], dtype=np.int64)
t0 = time.time()
for s in range(0, xb.shape[0], INSERT_BATCH):
    e = min(s + INSERT_BATCH, xb.shape[0])
    col.insert([ids[s:e].tolist(), xb[s:e].tolist()])
col.flush()
t1 = time.time()
print(f"insert_time_s={t1 - t0:.2f}")

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": NLIST},
}
t0 = time.time()
col.create_index(field_name="vec", index_params=index_params)
t1 = time.time()
print(f"index_build_time_s={t1 - t0:.2f}")
print("index_params:", col.indexes[0].params)

col.load()
print("\nMilvus IVF_FLAT results:")
for nprobe in NPROBES:
    search_params = {"metric_type": "L2", "params": {"nprobe": nprobe}}

    t0 = time.time()
    res = col.search(
        data=xq.tolist(),
        anns_field="vec",
        param=search_params,
        limit=K,
        output_fields=[],
    )
    t1 = time.time()
    qps = xq.shape[0] / (t1 - t0)

    pred = np.array([[hit.id for hit in row] for row in res], dtype=np.int64)
    r = recall_at_k(pred, gt, K)

    lat_ms = []
    for i in range(P99_SAMPLE):
        q = [xq[i].tolist()]
        s0 = time.time()
        _ = col.search(
            data=q,
            anns_field="vec",
            param=search_params,
            limit=K,
            output_fields=[],
        )
        s1 = time.time()
        lat_ms.append((s1 - s0) * 1000.0)
    p99 = float(np.percentile(lat_ms, 99))

    print(f"nprobe={nprobe:2d} qps={qps:8.1f} p99_ms={p99:7.2f} recall@{K}={r:.4f}")
