# One-pager: CAGRA on consumer AMD (follow-on)

**Audience:** DXC / AMD stakeholders  
**Date:** 2026-08  
**Status:** Harness + plan landed; lab triage next on `amd-rx7900xtx`

## Ask

Fund / schedule a **CAGRA follow-on** after the IVF_FLAT / IVF_PQ Milvus→HIP PoC,
targeting **consumer RDNA3** (RX 7900 XTX / `gfx1100`), not Instinct-only.

## Why

- Milvus GPU tutorials often lead with **`GPU_CAGRA`**.
- ROCm-DS / hipVS messaging emphasizes **Instinct**. Consumer cards are where
  many PoCs and SMB/workstation pilots actually run.
- We already proved sealed **GPU_IVF_FLAT / GPU_IVF_PQ** on gfx1100 with fair
  CUDA peer numbers. Graph ANN is the natural next index.

## What we know today

| Fact | Detail |
|------|--------|
| IVF PoC | Done (Layer-2 gate, Layer-3 smoke, Layer-4 parity) |
| CAGRA Catch2 | **FAIL** — recall **0.0** (TopK + Serialize, 2026-07-26) |
| Layer-2 gate | Still **PASS** (108 asserts; CAGRA not required) |

## Delivery plan

1. **Phase A (library, ~1.5–3 weeks):** reproduce → triage hipVS vs Knowhere →
   Catch2 CAGRA green → lib QPS/recall vs RTX 4080.
2. **Phase B (product, ~1–2 weeks after A):** sealed `GPU_CAGRA` smoke →
   Layer-4 SIFT (GIST if VRAM) → dashboard `amd-cagra-*` rows.

**Contingency:** if hipVS CAGRA is dead on gfx1100 after a 1-week time-box,
escalate minimal repro to AMD and pause B (no fake product green).

## Evidence package (repo)

- Plan/runbook: `docs/cagra_consumer_followon.md`
- Slides: `docs/porting_milvus_amd_slides.tex` (CAGRA frame)
- Scripts: `reproduce_cagra_gfx1100.sh`, `bench_cuvs_cagra.py`,
  `run_milvus_gpu_cagra_smoke.sh`, `run_milvus_layer4_cagra.sh`
- Status slot: `results/lib_bench/CAGRA_STATUS.md`

## Success looks like

- Catch2 CAGRA TopK + Serialize **PASS** on gfx1100.
- Lib and product recall@10 competitive with CUDA peer at matched params.
- Dashboard can show `amd-cagra` without painting IVF PoC red.
