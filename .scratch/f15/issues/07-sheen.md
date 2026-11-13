# 07: Sheen lobe and sheen albedo table

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_sheen` renders per decision 06 with the Charlie distribution and visibility and the base scaled by the sheen directional albedo; a Phosphor tool generates the E(μ, roughness) table (with a negative control) instead of importing the Sample Renderer PNG.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] Table generator reproducible (byte-identical twice) and checked against an independent numerical integration of the Charlie lobe; a corrupted table fails the check
- [ ] SheenTestGrid-like fixture (colour × roughness) passes decision-11 thresholds; furnace bound holds (base scaling prevents gain)
- [ ] Negative control (no albedo scaling) exceeds the furnace bound
- [ ] SheenTestGrid and one Tier B sheen model side by side under the declared rig
