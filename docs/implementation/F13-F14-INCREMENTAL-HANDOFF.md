# F13/F14 incremental source delivery — NON VERIFIED

Writer checkout: `/Users/danielsan/.codex/worktrees/f10-f12-development/phosphor`.
Branch: `codex/f13-f14-development`. The final delivery ref will be frozen only
when all authorized hooks and their source integration are present.

The initial immutable review snapshot is `2918f78` (selected by the tester);
`codex/f13-f14-review-base` also preserves `4e9773f49225f6c1593b6c2614e92f8f07b1cb6e`
with the first explicit diagnostic request fields. Use the tester's actual
integrated source as the baseline, preserving its owned corrections.

No configure/build, C++/MSL compiler, CPU/GPU test, renderer, profiler, simulator,
container, reference generation or performance measurement was executed by this
writer or its agents. Executed mechanical check: `git diff --check` only. Tests,
fixtures and serial runners are authored source, not execution evidence.

## Required integration preservation

The tester selected `2918f78`, merged it as `4b3303d`, then applied its owned
compiler fixes/gateway in `886ed00`. Preserve F9 compiler/archive routing and
`drainFrameReadbacks`, tester caster buckets, inherited ICB buffer10 textures,
F12 normalized receiver guides, and `f4629c8` receiver eye-facing normal,
shadow capture IDs3–5 and LIGHTING flushing. The tester also owns DI/GI chain-age
correction `970d7ce`; this writer does not modify that diagnosis or expiry.

The writer's `27c7b7e` carries only the non-PipelineCache source fixes from
`886ed00`, reconciled with the new owned F13 receiver/motion inputs. If already
present in the tester, preserve those fixes instead of applying them twice.
PipelineCache files and ROADMAP are untouched in this writer's complete range.
The typed gateway artifact remains in `docs/patches/f13-denoised-factory.patch`;
only the owner/tester applies and validates it. SDK output-unit policy remains
Unverified in production; an explicit fixture policy is an experiment.

## Incrementals after2918f78

- `4e9773f`: explicit F13/F14 diagnostics CLI requests and schema10 fields.
- `0780154`: isolated SDK constant/impulse/channels/lifecycle/wide-HDR requests.
- `017acce` (original agent `d9f545b`): actual foreign-view produced history,
  copied motion/normal negatives, pre-sanitization input checker and always-on
  per-signal history/moment checks; independent CPU fixtures written.
- `27c7b7e`: tester compiler corrections, preserving PipelineCache ownership.
- `fc9eb20`: consume recorded F13 numeric checks after completion on every frame,
  plus active-signal validation and authored test registration.
- `265d4f5`: explicit scalar capture from actual RGBA AO storage or R16/R32.

Further authorized F14 and native SDK hooks are being integrated as separate
commits. This document records the review base; it does not assert their delivery
until the final handoff lists their concrete files/APIs and frozen ref.

## Runtime and signal boundaries

Scene/RT preparation precedes geometric guides, DI/GI and base surface resolve.
F13 adds GGX-weighted specular Lo exactly once; AO affects residual ambient only
when GI is off. GI input to denoise/composition remains irradiance E and receives
albedo*(1-metallic)/pi exactly once. Raw capture IDs6/7 are specular/AO;3–5 stay
reserved for the tester's shadow diagnostics. Scalar PFM channels replicate R.

F14 supplies the same clock's sun/moon before all world-light/DI/GI producers,
uses exact geometry/material/environment epochs, propagates time-jump reset,
and composes atmosphere/fog/clouds before exposure/UI. Physical volume HDR stays
RGBA32Float. The cloud/air split is the declared centroid approximation, not an
exact arbitrary layered transport solution. Manual display exposure consumes
the approximate clock EV hint; histogram exposure meters actual physical HDR.

Custom denoise is the actual fallback for an unavailable native SDK/factory.
Native output bypasses ordinary F8 temporal reconstruction. Nonunit SDK
preExposure is blocked in production until the tester verifies its units.

Stop after F14. F14.5 weather remains unactivated pending real particle/material
consumers. No F15/OPT work, acceptance checkbox, push, PR or merge is performed.

Later source increments already present:

- `7def9f4`: tester463f2f4 fixture default-texture initialization.
- `c8eec66` / `026a17a`: legitimate internal noRT probe capture, external exact
  cooked input, supported expected-failure controls, valid nslog validation mode
  and frozen binary/source/metallib provenance in the serial runner.
- `90d6340`: malformed AO receiver guard before hardware rays or integer address
  conversion; input diagnostics remain independently recorded.

Raw AO storage is RGBA32Float in the actual F13 host. Scalar capture explicitly
extracts and replicates R, without changing its producer format or denoise input.

During integration, preserve both scalar selectors: shadow mask3 OR AO7.
The tester owns graph/parser selection for shadow3, world position4 and normal5;
the writer's specular6/AO7 additions must not replace those routes.
