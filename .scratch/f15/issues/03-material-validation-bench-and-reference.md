# 03: Material-validation bench and independent reference on the base model

Roadmap: F15.3

**What to build:** A material-validation bench with named scenarios (frozen light rig, camera, exposure, resolution), linear captures and offline-reference snapshots, an independent Python reference of the current base model written from the specification, a white-furnace scenario, and the F15 runner that applies the thresholds fixed in decision 11 with negative controls. Proves the harness before any new lobe.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Scenarios: base metallic × roughness grid under sun disk only, under constant environment only, furnace (environment radiance 1, albedo 1), extreme roughness
- [ ] Reference written from the glTF Appendix B formulas, independent of the MSL code (direct light exact, environment by quadrature; Mitsuba 3.9.1 only for ray casting where meshes need it)
- [ ] Runner reports per-fixture metrics against the decision-11 thresholds; direct-light grid passes p99 ≤ 1 %, max ≤ 3 %; environment grid mean error per sphere published (split-sum fit declared)
- [ ] Furnace: no pixel above 1.0 + 0.5 %; energy-loss table per roughness recorded
- [ ] Negative control per gated fixture (e.g. wrong roughness mapping, Fresnel off) fails its bound; a fixture whose control passes blocks the ticket
- [ ] Runner refuses to run concurrently with another GPU job and records the manifest (commit, asset hashes, seed, OS/SDK, power state)
