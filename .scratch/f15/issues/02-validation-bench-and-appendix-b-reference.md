# 02: Material-validation bench and independent glTF Appendix B reference

Roadmap: F15.3

**What to build:** A material-validation bench with named scenarios (frozen light rig, camera, exposure, resolution), linear captures and versioned offline-reference snapshots, an independent Python reference of the glTF 2.0 Appendix B model written from the specification (modelling the declared roughness clamp 0.04 and the sun-disk quadrature), a white-furnace scenario, and the F15 runner that applies the thresholds of decision 11 with negative controls and a GPU-job guard. Written test-first: on the current base model the intermediate-metallic grid and the furnace are expected to FAIL; ticket 03 turns them green.

**Blocked by:** None (can start immediately)

**Status:** resolved (verified 2026-11-13, e67d3d6/9848f3f/728ef15) with open criterion: runner lock refusal and manifest fields not evidenced in this ticket's record

- [x] Scenarios: base metallic × roughness grid under sun disk only, under constant environment only, furnace (environment radiance 1, albedo 1), extreme roughness
- [x] Reference independent of the MSL code (direct light exact, environment by quadrature); Mitsuba 3.9.1 only for ray casting, offline via `uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy`
- [x] Runner reports per-fixture metrics against decision 11; on the current code metallic ∈ {0, 1} direct grids pass, metallic ∈ (0, 1) and the furnace fail, and the failure is recorded as the expected red state
- [x] Negative controls (e.g. wrong roughness mapping, Fresnel off) fail their bounds; a fixture whose control passes blocks the ticket
- [ ] Runner refuses to start while another GPU job runs (lock + process check) and records the manifest (commit, asset hashes, seed, OS/SDK, power state) — PARTIAL (audit 2026-11-15): per-capture GPU guard and SHA-checked cache recorded (75a055c, e4fe702, b14d44d); no test of the refusal nor the manifest fields quoted

## Comments

- 2026-11-13 (orchestrator): written by a Sonnet subagent (commits ee9e368,
  e09e325, b14d44d, e4fe702), reviewed and integrated. GPU run on e67d3d6:
  `tools/f15_check.py --run --expect red` → `EXPECTED_RED_CONFIRMED` after
  the terminator-band ROI rule (decision 11 amendment): direct grid passes at
  metallic 0 (p99 0.08 %, max 0.69 %) and 1 (p99 0.09 %, max 0.43 %), fails at
  0.25/0.5/0.75 (p99 19.8/39.5/58.5 %); environment grid within 5 % per sphere
  (the missing diffuse Fresnel weighting moves means by ≤ 4.4 %, gated by the
  furnace instead); furnace max 1.428 (limit 1.005) with the reference's own
  literal-quadrature maximum 1.372 recorded; every negative control fails its
  bound. Captures in `build/f15-check/t02-red/`.
- Finding for ticket 03: the literal Appendix B dielectric is not energy
  conserving at grazing views (diffuse weighted by the incident-direction
  Fresnel while the specular lobe goes to 1); the engine's split-sum weighting
  `1 − E_spec` keeps the furnace bound, the reference maximum is reported, not
  gated.
- 2026-11-13 (orchestrator): code review (standards + spec axes, fresh
  reviewers) of tickets 01/02/04; every finding fixed by a Sonnet implementer
  (12 commits, integrated 15:00) or closed in ticket 03 (shared-module
  duplicates). Verified on M5 after integration: build, ctest 15/15
  (phosphor_tests 768 cases), F13-path exact captures byte-identical to the
  post-ticket-03 baseline, visual_check 8/8 at 0 px vs
  `build/reference-f15-appB`, `tools/f15_check.py --run --expect green` →
  GREEN on 7 scenarios incl. `base-grid-env-resolve` and
  `roughness-extremes-sun`; declared-model gates: worst sphere mean 0.009 %
  (bound 0.5 %), worst p99 0.037 % (bound 1 %), the three declared controls
  (diffuse unweighted, Karis fit, F0-blended diffuse weight) fail. Captures
  `build/f15-check/review`.

2026-11-15 — closure audit (ticket 24):

- 1 scenarios (sun grid, env grid, furnace, extremes): MET (9848f3f; GREEN on 7 scenarios at 2e1057a). Ticked.
- 2 reference independent of MSL: MET by construction (e09e325, 728ef15; tests/test_f15_reference.py 36 tests). Ticked.
- 3 runner metrics and expected red: MET (e67d3d6, EXPECTED_RED_CONFIRMED; p99 0.08/0.09 % at metallic 0/1, 19.8-58.5 % intermediate, furnace 1.428). Ticked.
- 4 negative controls fail: MET (every control fails in the red run; three declared controls fail post-review). Ticked.
- 5 runner lock refusal + manifest: PARTIAL (guard and cache recorded in 75a055c/e4fe702/b14d44d; no refusal test or manifest fields in the record). Open.
