# 10: Sheen lobe, sheen albedo table and its environment term

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_sheen` per decision 06: Charlie distribution and visibility, base scaled by the sheen directional albedo from a Phosphor-generated E(μ, roughness) table, and the sheen environment term in the composition (probe/GI radiance at N × Charlie albedo).

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Table generator reproducible and checked against an independent numerical integration; a corrupted table fails the check
- [ ] Sheen grid (colour × roughness) passes decision-11 thresholds under constant environment and under a probe-lit fixture; furnace bound holds
- [ ] Negative control (no albedo scaling) exceeds the furnace bound
- [ ] SheenTestGrid and one Tier B sheen model side by side
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
