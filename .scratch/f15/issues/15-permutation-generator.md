# 15: Material permutation generator and budgets

Roadmap: F15.2

**What to build:** The app exports feature-class manifests from fixtures and corpus; the generator (extension of `variant_gen`) turns manifests plus a declared list into the class table, MSL constants and pipeline descriptors and updates the archive in a targeted way; per-class tile binning is measured against the union pipeline and adopted by the ≥ 5 % rule; budgets are enforced and reported.

**Blocked by:** 13, 14

**Status:** ready-for-agent

- [ ] Generator determinism: two runs byte-identical (ctest); malformed manifests and schema errors rejected with messages
- [ ] Every generated class pipeline equals the generic resolve exactly (0 pixels) on every fixture; negative control (forcing a class missing a used lobe) differs
- [ ] Budget: ≤ 64 archived material permutations, ≤ 16 active classes per scene (beyond: logged generic fallback), archive build time +≤ 25 % over the measured baseline, 0 render-thread compiles and 0 fallback draws in measured frames (`tools/archive_check.sh`, `tools/hitch_check.sh`)
- [ ] Binning vs union measured on the chess fixture (A/B/A, 3 replicas, quiet machine); decision and numbers in `docs/opt-log.md`/`docs/perf-log.md`
- [ ] Targeted archive update (no full re-harvest duplicates); `pipelines script` test green
