# F10–F14 — active integration handoff, 2026-10-07

Owner scope: develop, verify and integrate through F14, then STOP. No F15/OPT.
F9 is merged in PR18/main d5ec450. F10–F14 are NOT yet accepted or merged.
Active checkout: `/Users/danielsan/.codex/worktrees/f9-core/phosphor`, branch
`codex/lighting-integration`, Release build `build/lighting`.
Only the root tester runs GPU work, one process at a time. Source writer chat
“Sviluppo Phosphor” supplied frozen41ef803 and SDK prewarm c42b814 (integrated).
Physical M5 Max128GB/macOS27.2; forcedApple9 is not physicalM3 certification.

## Established evidence

- F10 physical shadow oracle v2 PASS: unchanged policy, CSM/RT penumbra width
  errors2–4%, flicker and physical bias negative pass. Fixes9d49e21/838f446.
- F10 alpha/lifecycle v5 PASS: all guide AND mask comparisons exactly equal
  indexed/mesh at100/75/50%, native/effectiveApple9; mip negative sensitive;
  1–4views, DRS, resize/reset pass. Fixes89baff6,16359a7,b3c869d.
- F11 before temporal reuse:22 failed statistical gates. Chain-age expiry fix
  970d7ce has exact independent rational proof. Corrected9cases/459comparisons
  PASS on32candidate+32reference seeds. Point1 exact and PDFx2=0.5 negative
  PASS. Source: docs/research/2026-10-07-f11-energy-chain-age-validation.md.
- F12 CPU oracle validated independently: Cornell65536spp x2seeds; thin wall
  16384spp x2seeds. Primary geometry/UV/emission/signal roles verified.
- F12 receiver depth-unprojection produced primary self-hits. Exact primitive
  reconstruction eliminates them (41171d0/b9174f2); indexed canonicalized too.
- F12 raw: DDGI16x8x16 and ReSTIRbase pass static6ROI; raw cache fails noise.
  All4 custom-filtered cases PASS unchanged six-ROI20%bias/30%NRMSE gates.
- Thin-wall candidate passes; disabling actual moment visibility preserves
  invariant PASS but fails independent leakp95 (25.669%>25%). Positivep957.92%.
- Le12->6 step snapshots255/256 prove unchanged geometry/camera and scaled
  emission. DDGI/custom16x8x16 recovers stably by87frames; cache dense by124;
  ReSTIRbase FAIL31.13% at128. Added denseReSTIR passes by91, with subsequent
  outlier frames. DDGI dense is the proposed full-GI baseline pending costs.
- F13 mutable VisibilityRenderer::replaceColor retargeted earlier resolve
  callback writes into future HDR memory aliased with motion.690e18a separates
  composed selection from immutable resolve targets. Same GPU4frame alias/
  noalias captures now byte-identical. No barrier widening was needed.
- F13 functional matrix32frames (mirror/roughness/raw/custom/SSR/AO/probe
  capture and authored negatives) PASS; not whole temporal quality acceptance.
- F13 SDK initial async request used to finish after the fixture ended.
  Prewarm48ebc2b fixes it. Constant-unit96actual nativeframes PASS; constant
  preExposure pair, impulse pair and actual guide contracts PASS.
- SDK channels had an unannounced wrap64: original ghost/recovery FAIL remains.
  f692593 signals the known source change; unchanged caps now PASS, recovery0
  vs8, frames0–63 byte-identical. Protocols/results committed.
- F14 sparse numerical tests PASS for zenith/horizon/space, homogeneous fog,
  and controls associated with clouds. Corrected partial LUT epoch writes in
  d4b712a. Units/light/stale-LUT/foreign eligible history negatives detected.
  Freeze-clock controls6f3fe7e avoid fabricated history eligibility.

## Current blockers / next work

1. SDK extreme colored HDR [368640,128,64] is NOT accepted. q=1/64 alone loses
   energy; autoExposure saturates. Manual exposure hint recovers red but tiny
   channels remain inaccurate. Native R16 texture rounding produced22*2^-24
   instead of CPU RNE23; next exact-representable experiment6d68286 supplies23
   explicitly. Target/q/restore/gates unchanged. Production policy unpromoted.
2. Reviewer is preparing true Float32 lighting residuals: current visibility
   target and indexed lit fragment half output can overflow BEFORE F13/F14
   Float32 composition. Preserve legacy F7/F8; audit post/upscaler format
   contracts and make unsupported combinations explicit, never silent clipping.
3. Finish actual SDK lifecycle/resize/four-view/hot-reload/leaks tests and
   remaining integrated temporal scenarios. Constant/guide fixtures alone do
   not certify arbitrary moving-light GI/reflection quality.
4. Benchmark GI presets and full representative frames x3 on a quiet machine,
   no captures/debug validation, check O7 steady allocations. Adopt defaults
   only with declared quality/cost. Current validation timings are diagnostic.
5. Full F9/F6/F7/F8 regression, archive harvest/hot reload/leaks, residue audit,
   documentation/task ticks only with evidence; PR/CI/merge into main last.

## Local evidence and collaborators

Root artifacts: `build/f10-quality-oracle-v2`, `build/f10-quality-alpha-life-v5`,
`build/f11-energy-reuse-chainage-v2`, `build/f11-energy-point1-chainage`,
`build/f12-reference-candidates-v3`, `build/f12-filtered-candidates-v1`,
`build/f12-leak-recovery-v1`, `build/f12-restir-dense-recovery-v1`,
`build/f13-frozen-resolve-ab`, `build/f13-functional-final-v1`,
`build/f13-native-constant-v2`, `build/f13-native-units-v1`,
`build/f13-wide-auto-v1`, `build/f13-wide-manual-v1`,
`build/f13-wide-manual-exact-v1`, `build/f13-channels-wrap-v1`,
`build/f14-numerical-positive-v1`, `build/f14-negative-controls-v1`.
Failures remain preserved. The controlled SDK experiments are not production
policies. No stale source-only handoff supersedes actual current evidence.

f9_proxy owns CPU oracle/quality analysis in the f9-proxy worktree; f9_review
owns isolated residual-HDR and SDK analysis patches OUTSIDE this checkout;
f9_shaders provided F10/F14 fixes and SDK temporal analysis. They do not run
GPU. Writer chat id01a11660-e6e3-7f73-977a-07fa754bfd08 is source-only.
