# Shared gtest timing results (management lib compare)

JSON from `scripts/run_hipvs_gtest_timing.sh` / `run_cuvs_gtest_timing.sh`.

See [hipvs_vs_cuvs_gtest_timing.md](../docs/hipvs_vs_cuvs_gtest_timing.md).

After both sides exist:

```bash
python3 scripts/compare_gtest_timings.py \
  --hipvs results/gtest_timing/gtest_timing_hipvs_....json \
  --cuvs  results/gtest_timing/gtest_timing_cuvs_....json \
  --out-md results/gtest_timing/COMPARE.md
```
