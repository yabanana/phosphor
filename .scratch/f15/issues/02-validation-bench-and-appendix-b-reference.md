# 02: Material-validation bench and independent glTF Appendix B reference

Roadmap: F15.3

**What to build:** A material-validation bench with named scenarios (frozen light rig, camera, exposure, resolution), linear captures and versioned offline-reference snapshots, an independent Python reference of the glTF 2.0 Appendix B model written from the specification (modelling the declared roughness clamp 0.04 and the sun-disk quadrature), a white-furnace scenario, and the F15 runner that applies the thresholds of decision 11 with negative controls and a GPU-job guard. Written test-first: on the current base model the intermediate-metallic grid and the furnace are expected to FAIL; ticket 03 turns them green.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Scenarios: base metallic × roughness grid under sun disk only, under constant environment only, furnace (environment radiance 1, albedo 1), extreme roughness
- [ ] Reference independent of the MSL code (direct light exact, environment by quadrature); Mitsuba 3.9.1 only for ray casting, offline via `uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy`
- [ ] Runner reports per-fixture metrics against decision 11; on the current code metallic ∈ {0, 1} direct grids pass, metallic ∈ (0, 1) and the furnace fail, and the failure is recorded as the expected red state
- [ ] Negative controls (e.g. wrong roughness mapping, Fresnel off) fail their bounds; a fixture whose control passes blocks the ticket
- [ ] Runner refuses to start while another GPU job runs (lock + process check) and records the manifest (commit, asset hashes, seed, OS/SDK, power state)
