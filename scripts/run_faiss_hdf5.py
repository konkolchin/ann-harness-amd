import time
import h5py
import numpy as np
import faiss

K = 10
DATA_PATH = "data/sift-128-euclidean.hdf5"


def recall_at_k(pred, gt, k):
    return np.mean(
        [len(set(pred[i, :k]).intersection(set(gt[i, :k]))) / k for i in range(pred.shape[0])]
    )


with h5py.File(DATA_PATH, "r") as f:
    xb = np.array(f["train"], dtype="float32")
    xq = np.array(f["test"], dtype="float32")
    gt = np.array(f["neighbors"], dtype="int64")

d = xb.shape[1]
print(f"xb={xb.shape}, xq={xq.shape}, gt={gt.shape}, dim={d}")

# Exact baseline
flat = faiss.IndexFlatL2(d)
flat.add(xb)
t0 = time.time()
_, _ = flat.search(xq, K)
t1 = time.time()
print(f"[Flat] qps={xq.shape[0] / (t1 - t0):.1f}, recall@{K}=1.0000")

# IVF baseline
nlist = 1024
ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
ivf.train(xb[:200000])
ivf.add(xb)

for nprobe in [1, 4, 8, 16, 32]:
    ivf.nprobe = nprobe
    t0 = time.time()
    _, I = ivf.search(xq, K)
    t1 = time.time()
    r = recall_at_k(I, gt, K)
    print(f"[IVF] nprobe={nprobe:2d} qps={xq.shape[0] / (t1 - t0):8.1f} recall@{K}={r:.4f}")
