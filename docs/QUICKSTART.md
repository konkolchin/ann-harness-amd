# Quickstart

This is the shortest reproducible flow for a new user.

## 0) System prerequisites (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
```

## 1) Clone repository

```bash
git clone https://github.com/konkolchin/ann-harness-amd.git
cd ann-harness-amd
```

## 2) Create and activate Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Download dataset

```bash
mkdir -p data
wget --no-proxy https://ann-benchmarks.com/sift-128-euclidean.hdf5 -O data/sift-128-euclidean.hdf5
```

## 4) Optional dataset sanity check

```bash
python - <<'PY'
import h5py
f = h5py.File("data/sift-128-euclidean.hdf5", "r")
print(list(f.keys()))
for k in f.keys():
    print(k, f[k].shape, f[k].dtype)
PY
```

Expected keys:

- `train`
- `test`
- `neighbors`
- `distances`

## 5) Run FAISS baseline

```bash
python scripts/run_faiss_hdf5.py
```

## 6) Run Milvus locally (Milvus Lite — smoke test only)

**Milvus Lite** ships with `pip install -r requirements.txt` (`pymilvus[milvus_lite]`).
No separate server download. Data goes to `./milvus_sift.db`:

```bash
python scripts/run_milvus_hdf5.py
```

Notes:

- Good for: insert/index/search smoke test, QPS/p99 plumbing, pymilvus API check
- **Not valid for `nprobe` sweeps** — see runbook Table 2 (PDF): recall@10 stays flat
  (~0.894) while FAISS rises from ~0.37 to ~0.98 across the same sweep
- For production-like Milvus + working `nprobe`, use standalone server (sections 7.1–7.2)

Optional: point scripts at a full Milvus server instead of Lite:

```bash
python scripts/run_milvus_hdf5.py --uri "http://HOST:19530"
```

## 7) Diagnose Milvus nprobe behavior

```bash
python scripts/investigate_milvus_nprobe.py
```

Or against standalone Milvus:

```bash
python scripts/investigate_milvus_nprobe.py --uri "http://localhost:19530"
```

This script prints:

- FAISS reference curve
- Milvus curves
- digest variability summary (whether `nprobe` changes returned neighbors)

## 7.1) Full Milvus server in Docker (needs 8 GB+ RAM)

Standalone Milvus is **required** to validate `nprobe` after Table 2 Lite results.
There is no lighter “full Milvus” binary for local use — only Docker (or a managed
cloud cluster).

**Do not run this on Hongkong-Ubu20 if it OOM'd or dropped SSH.** Use section 7.2
instead: start the server on a machine with enough RAM; run harness scripts as client.

**Resource warning:** embedded etcd + Milvus often needs **8 GB+ RAM**. Small VMs hang
at “Wait for Milvus Starting…”, hit the OOM killer, or drop SSH.

### What `standalone_embed` means

Milvus **standalone** = one Milvus node (not a distributed cluster).

**embed** = **embedded etcd**: etcd (Milvus metadata store) runs **inside the same
Docker container** as Milvus. You do not start separate `etcd` / `minio` containers.

`standalone_embed.sh` is Milvus’s official helper. It:

- writes `embedEtcd.yaml` (etcd listen/advertise URLs inside the container)
- writes `user.yaml` (optional config overrides)
- creates `volumes/milvus/` for persistent data
- runs `docker` with the env vars and mounts embedded etcd needs

A bare `docker run … milvus run standalone` often exits immediately with
`panic: context deadline exceeded` because embedded etcd is not configured.

### Start (recommended)

Requires Docker. From the repository root:

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

The script waits until the container is healthy (often 60–90 seconds). Verify:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep milvus
curl -sf http://127.0.0.1:9091/healthz && echo " OK"
```

Then use `--uri "http://127.0.0.1:19530"` in Milvus scripts.

Stop or remove later:

```bash
bash standalone_embed.sh stop
bash standalone_embed.sh delete   # removes container + local volumes/config files
```

## 7.2) Split setup: server elsewhere, harness on AMD box

When the benchmark host is too small for Docker Milvus:

1. **On a host with 8 GB+ RAM** (WSL2, laptop, cloud VM, RX 9070 box): section 7.1
2. **Open port 19530** to the AMD machine (firewall / security group), or use SSH tunnel:

```bash
# On the AMD box — forward remote Milvus to local 19530
ssh -N -L 19530:127.0.0.1:19530 USER@MILVUS_HOST
```

3. **On the AMD box** (venv + dataset only; no Docker Milvus):

```bash
source .venv/bin/activate
python scripts/investigate_milvus_nprobe.py --uri "http://127.0.0.1:19530"
python scripts/run_milvus_hdf5.py --uri "http://127.0.0.1:19530"
```

Replace `127.0.0.1` with `MILVUS_HOST` if the server is reachable on the network directly.

**Until standalone is available:** treat **FAISS** (runbook Table 1) as the ground
truth for recall-vs-`nprobe` tradeoffs. Table 2 (Milvus Lite) documents connectivity
only, not ANN tuning behavior.

## 8) Common issues

### Download blocked by proxy/firewall

Try `--no-proxy` (already used above). If still blocked, download on another machine and copy file into `data/`.

### Milvus insert error: message too large

Reduce `INSERT_BATCH` in `scripts/run_milvus_hdf5.py` (e.g. `20000`).

### Milvus Lite: recall@10 unchanged across nprobe (expected)

This matches runbook **Table 2**: flat recall (~0.894) while FAISS (Table 1) rises
with `nprobe`. **Do not use Lite results for ANN tuning comparisons.**

Next step: standalone Milvus (sections 7.1–7.2). Lite cannot substitute.

### Milvus Docker: `context deadline exceeded` on startup

The container was created but exited because embedded etcd did not become ready.
Use `standalone_embed.sh` (section 7.1), not a one-line `docker run`.

If it still fails:

```bash
docker logs --tail 50 milvus-standalone
free -h    # standalone Milvus wants several GB RAM
df -h .    # slow or full disk can make etcd time out
env | grep -i proxy
```

If `http_proxy` / `https_proxy` are set, unset them and restart:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
bash standalone_embed.sh start
```

### Wrong directory

Run commands from repository root:

```bash
ann-harness-amd/
```
