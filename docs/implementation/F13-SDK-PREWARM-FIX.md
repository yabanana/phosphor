# F13 SDK fixture initial prewarm fix — SOURCE ONLY / NON VERIFICATA

Assigned after tester cold-model run completed96 fixture frames before the
async denoised factory future became ready. The writer ran no configure/build,
C++/MSL compile, CPU/GPU test, renderer or profiler. Mechanical check only:
`git diff --check`. Initial frozen delivery `41ef803` is not rewritten.
Branch: `codex/f13-sdk-prewarm-fix` in the existing writer worktree.

The separate carry commit `c1a2029` contains the tester's existing `5ae3d15`
FXDepthOutput/filtered-GI option compile corrections. Skip it if already present.
Apply the prewarm fix itself on the tester's current integrated source, preserving
its owned PipelineCache gateway, live F14 stamp read/write dependency correction,
exact receivers, alias/motion diagnosis and all other root fixes.

## Concrete behavior

The fixture's first REAL prepare occurs inside an already acquired Metal frame,
with actual input/output extents and canonical scenario/preExposure/48-frame phase.
Fixture-only `resizeSettleFrames=0` schedules the first exact descriptor immediately.
Production adapter options retain asynchronous settling/fallback behavior.

`MetalfxDenoise::prewarmPreparedFixture(activeViews,budget)` queues every actual
active view against the same complete extent/generation/flags key. Their future
waits share ONE steady-clock deadline. No repeated prepareFrame, synthetic frame,
phase/index increment, arbitrary sleep, new compiler, ordinary-scaler cast or
unbounded waitAllFinal/collect(true) is used. Ready collection keeps existing
obsolete-result, format and usage checks. Only the prepared frame's status and
pack/restore uniforms are published again; history.begin is not repeated.

The CLI `--denoised-fixture-prewarm-ms` accepts1..120000, default120000ms. It is
fixture-only. Missing SDK/device/factory and invalid/unverified output policy
return their terminal diagnostic immediately, with no native evidence. A pending
request timeout/superseded key fails BEFORE counting/encoding the requested frames.
Neither the wait nor its diagnostic cancels the underlying initialization.

A new immutable `prewarm.json` records actual initial extent/view count, frame,
phase, generation, budget/duration/reason and zero native encodes before counting.
The completed SDK summary includes the initial prewarm result. The native runner
requires completed prewarm, matching first native extent, unchanged frame0/phase0,
all actual active views and the exact frozen per-case budget. Existing native
encoded/readback/packed-channel/PFM equations remain mandatory and unchanged.

For warmup0, Engine initializes measurement/feedback/allocation baselines AFTER
initial fixture prewarm and BEFORE the same frame0 is submitted. Startup elapsed
is excluded from first-frame CPU/prepare metrics and the next SDL time interval,
without changing Metal frame indices, requested frame count or exposure phases.
Actual prewarm duration remains explicitly reported as startup, not hidden.

The planner freezes this budget into manifest/case/command. Replay validates the
outer process timeout against the actual frozen budget after loading the plan;
a CLI default cannot override it. Written counterexample tests cover frozen120s
with outer2s (reject), frozen60s with outer90s (accept), and altered case commands.
These tests, and the bounded CLI tests, were authored but NOT EXECUTED.

## Tester validation — NOT EXECUTED HERE

After the owned gateway build, freeze a NEW plan (older manifests did not contain
the prewarm contract), then run the same constant96frames128x96 case. Verify actual
ready prewarm, frame0/phase0, one requested extent, then nonzero real native encodes,
96 actual output/readback tags and existing unit/constant/channel gates. Preserve
raw prior failure. No threshold change or automatic production-policy promotion.

Exercise a short budget separately to confirm explicit initial timeout and zero
counted native frames. The outer process timeout remains essential: existing
adapter destruction/queue worker join may still wait for uncancelled initialization.
This fix covers INITIAL requests, including all active views. Later lifecycle
resize/reload requests retain the fixture's existing async behavior; unexercised
transitions remain failed/unverified, never waived. No SDK lifetime acceptance or
F13/F14 roadmap tick follows from this source patch. Stop after F14 remains in force.
