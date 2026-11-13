# 21: Material permutation generator and budgets

Roadmap: F15.2

**What to build:** The app exports feature-class manifests from fixtures and corpus; the generator (extension of `variant_gen`) turns manifests plus a declared list into the class table, MSL constants and pipeline descriptors and updates the archive in a targeted way; per-class tile binning measured against the union pipeline on the chess fixture (provisional decision; final with the terrain in ticket 23); budgets enforced and reported.

**Blocked by:** 19, 20

**Status:** ready-for-agent

- [ ] Generator determinism: two runs byte-identical (ctest); malformed manifests and schema errors rejected
- [ ] Every generated class pipeline equals the generic all-lobes resolve exactly (0 pixels) on every fixture; negative control (class missing a used lobe) differs
- [ ] Budget: ≤ 64 archived material permutations, ≤ 16 active classes per scene (beyond: logged generic fallback), archive build time +≤ 25 %, 0 render-thread compiles and 0 fallback draws in measured frames
- [ ] Binning vs union on the chess fixture (A/B/A, 3 replicas, quiet machine) recorded as provisional in opt-log/perf-log
- [ ] Targeted archive update; `pipelines script` test green
