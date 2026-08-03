# CAGRA lib / Catch2 status (gfx1100)

## Baseline (pre-follow-on)

| Date | Suite | Result |
|------|-------|--------|
| 2026-07-26 | Knowhere Catch2 CAGRA TopK / Serialize | **FAIL** recall **0.0** |
| 2026-07-26 | Layer-2 gate `Test Gpu Index Search` | PASS (108) |

## Lab fills (after `reproduce_cagra_gfx1100.sh` / benches)

| Artifact | Path | Result |
|----------|------|--------|
| Catch2 summary | `$WORKDIR/logs/cagra_repro_summary_*.md` | *pending lab* |
| hipVS lib JSON | `$WORKDIR/logs/lib_hipvs_cagra_*.json` | *pending lab* |
| cuVS lib JSON | `~/milvus_cuda_4080/logs/lib_cuvs_cagra_*.json` | *pending lab* |
| Compare table | paste below | *pending* |

### Compare table (fill)

```
itopk   hipVS QPS   cuVS QPS   speed-up   R@10 hip   R@10 cu
```

## Triage conclusion (fill)

- [ ] hipVS-only broken on gfx1100 → upstream hipVS / ROCm-DS
- [ ] hipVS OK, Knowhere broken → patch under `patches/knowhere/` (0052+)
- [ ] Serialize-only → product load blocker for Phase B
