# F10–F14: integration and measured boundaries

The renderer implementations and diagnostic tools are integrated in
`codex/lighting-integration`, on top of F9/PR18. This is an **experimental
development baseline**, not completion of every phase exit criterion.
The owner stop remains F14: no F15 or new OPT implementation is activated.

Hardware: Apple M5 Max, 128 GB, macOS27.2 (26B5091g). Forced Apple9 exercises
capabilities on this same device; it does not certify a physical M3 or T0.
The compact [evidence ledger](results/F10-F14-integration-M5Max-2026-10-07.json)
preserves scope, commands, hashes, results and failed gates. Raw captures and
logs remain in the aggregator worktree's `build/` directories.

## Delivered scope and acceptance

| Tasks | Implemented and exercised | Remaining acceptance boundary |
|---|---|---|
| F10.1–F10.2 | Mesh/indexed CSM/PCSS and solar RT; independent penumbra oracle, bias/flicker controls | Representative measured fixtures, not all scene/light configurations |
| F10.3 | Selected local-light visibility used by ReSTIR DI | Broad moving-light image quality remains separate from estimator checks |
| F10.4–F10.5 | Bounded contact shadows and regional static-shadow cache; runtime checks and corruption controls | Contact-edge visual coverage beyond the fixtures is not certified |
| F11.1–F11.3 | Candidate/temporal/spatial ReSTIR DI, area/textured emitters, realized alias PDFs | Dynamic image-quality and product frame budgets remain open |
| F11.4 | STBN consumer verified bit-exact and against frozen distribution/correlation checks | Distribution evidence is not rendered-image quality evidence |
| F11.5 | Reduced clustered/DI path and Many Lights workload | Physical base/Pro devices unavailable; 1024-light full profile is about25 ms on M5 Max |
| F12.1–F12.5 | DDGI, keyed radiance cache, ReSTIR GI, emissive changes, independent Mitsuba export/reference | Dense DDGI is the strongest measured quality candidate; cost gate and Sponza dynamic quality remain open. Cache/ReSTIR variability is retained as a limitation |
| F13.1–F13.2 | RT/SSR/cache/probe reflection paths, RTAO/GTAO, separate signal composition | Broad roughness/disocclusion references remain pending |
| F13.3 | Typed native adapter and real SDK channel/lifecycle fixtures | **Partial:** production native radiometry is unqualified and uses custom Float32 before the graph. Native SDK retains640 B per creation in the standalone reduction |
| F13.4 | Per-signal temporal/atrous custom denoising | Static Sponza still shows specular noise with frozen clock; AO/filtered-GI successes do not close this quality defect |
| F13.5 | Static raster/RT/cooked cubemaps, hierarchy bounds, GGX prefilter, parallax; wide HDR preserved on supported Apple9/reduced paths | The bounded split-sum BRDF remains approximate, especially rough/grazing cases |
| F13.6 | Immutable linear captures, independent physical AO control, history checkpoints, temporal metrics and negative controls | **Partial:** moving-camera disocclusions and a converged dynamic GGX/reference corpus remain pending |
| F14.1–F14.4 | Atmosphere LUTs, aerial perspective, froxel fog, clouds, sun/moon/stars and clock jumps | Sparse physical numerics and300-frame full-rate comparisons pass. Continuous celestial changes conservatively reset fog/cloud history; general temporal reuse under changing radiometry is not claimed |
| F14.5 | Nonactivated weather candidate | Deliberately deferred to the particle/material dependencies; not required to add F15/OPT now |

No unchecked roadmap task is implicitly completed by this source integration.
The detailed [F12 status](implementation/F12-STATUS-2026-10-07.md) retains its
per-task evidence and gaps. Native admission follows the
[F13 radiometry policy](implementation/F13-RADIOMETRY-POLICY.md).

The final static-Sponza visual check exposes substantial specular noise, also
with the celestial clock frozen and auto exposure enabled. Completed GPU
checkpoints at640×360 show full valid/reused histories for DI, GI and AO by
frame31; specular reuse is110008/230400 pixels (47.75%) in that frame. Thus a
global history reset is not established as the sole cause. Per-ray path/hit
rejection in the specular filter is a concrete candidate for further bounded
verification, not a proved or corrected cause. F13.4/F13.6 qualification
remains open. The opt-in history diagnostic changes no denoise parameters or
shader algorithm and requires `--debug-lighting`.
Raw evidence: `build/f13-history-sponza-static-v1/history.json`,
`build/f10-f14-sponza-daylight-review/sponza-daylight.png` and
`build/f10-f14-sponza-frozen-review/sponza-frozen.png` in the aggregator checkout;
hashes and frame31 checkpoints are in the integration ledger.

## Independent results

- F10: unchanged physical penumbra gates pass for CSM and RT; alpha receiver
  position, normal and mask are identical between mesh and indexed overflow
  at100/75/50%, including forced Apple9. The mip negative remains sensitive.
- F11: the original temporal-expiry implementation failed22 statistical gates.
  The chain-age correction passes all459 comparisons across9 cases with the
  original seed sets and confidence limits. The PDF×2 negative produces the
  expected half-energy result. This does not prove zero population bias.
- F12: independently validated Cornell and thin-wall references, filtered GI
  comparisons and the real Le12→6 step. DDGI/custom16×8×16 recovers by87 frames
  and stays within the retained tail gates; reduced-density/default, cache and
  ReSTIR results are not silently substituted by this successful configuration.
- F13 AO: the64-frame geometric wall-step test passes the analytic projected
  disk reference. RMSE p95=0.0056366, max=0.0066108, flicker=0.00025965,
  ghost=0 and persistent recovery=0 frames. Actual GPU history grows to32,
  resets to1 at the geometry step, then resumes reuse. This is an AO result.
- F14: all five300-frame volume runs pass functional and sparse numerical
  checks. Reconstructed/full-rate cloud comparison passes retained thresholds:
  p95=0.00216393, flicker=0.000268225, ghost=0.0213886, cut recovery=0.
  The full-rate control is not relabeled as an independent offline renderer.

## Integration defects corrected

Corrections include unbiased history-chain expiry, physical CSM caster bounds,
exact primary receiver reconstruction, matching alpha footprints for lighting,
independent immutable resolve/composed-color targets, Float32 physical residuals
and probe storage, correct partial LUT epoch dependencies, real clock/history
negatives, and exposure diagnostics using the actual Post input.

A legacy alpha regression was caught by the F7/F8 held-out test. The preserved
F9 forward image was unchanged, while87 visibility pixels had changed. The
legacy entry point is now preserved separately from the lighting receiver
entry. Held-out PSNR returns to the F9 value50.6502559 dB, with unchanged caps.

Archive harvest originally discarded unlabeled pipeline descriptors and then
exposed duplicated function labels across compiler library IDs. The merger now
preserves complete pipeline identities, validates metadata and canonicalizes
only engine-library references. Depth-only, HDR, alpha and linked RT coverage
are tested. Positive hot reload and the actual changed-alpha negative both
pass with the active archive; private SDK libraries are never relabeled.

## Lifetime and compatibility

The custom full-frame path passes128 frames, four views, cuts and resize with
zero at-exit leaks in its production configuration. With API/shader validation,
the same workload reports a2128 B compiler-cache allocation. The review found
balanced application ownership; validation also disables archive lookups, so
the cause is **not attributed** to either application or driver. That run
remains failed and visible in the ledger.

The native **denoised** SDK is distinct from the accepted F8 temporal path.
Serial construction fixes the observed shared-compiler initialization spin;
bounded fixture-only waits make resize/reload coverage real. All300 native
frames, four views, three extents and two generations pass their channel and
numeric checks. At-exit still fails:20×640 B. Public API create-only probes
with1/4 scalers reproduce640/2560 B, with and without validation, while weak
references prove wrapper destruction. The [reduction](../bench/metalfx_denoised/sdk_create_only.mm)
and [results](results/F13-SDK-create-only-M5Max-2026-10-07.json) preserve this
boundary. No F8 private ownership workaround is applied to the denoised SDK.

Native wide-colored-HDR experiments also fail the original1% radiometry gate.
That parameter search is stopped. Requested native/custom production captures
are byte-identical on the guarded path, with zero SDK requests/encodes and no
native packing branch. Standard F8 temporal reconstruction with the new
Float32 lighting chain is explicitly unsupported; native/spatial post works.
Ordinary legacy F7/F8 temporal regression remains green.

## Measured cost and decision

Three paired A/B/A replicates per scene,1920×1080,128 warmup+512 measured frames,
fixed timestep, native post, serial GPU timing, no validation/capture/checkers:

| Incremental DDGI/custom16×8×16 cost | Median paired p50 delta | Range across3 replicates |
|---|---:|---:|
| Cornell |3.0087 ms|2.97505–3.01365 ms|
| Sponza |6.2240 ms|6.18665–6.8446 ms|

All18 runs have zero measured GPU allocations; all six pairs pass the frozen
drift gates. The analyzer's original native enum was wrong: the real report
uses `native-spatial`. Reanalysis additionally requires640 native/zero temporal
frames and native input dimensions. The failed first analysis is retained;
no timings, captures or thresholds were changed.

Separate full-profile runs use128 warmup+256 measured frames and three replicas:

|1080p profile on M5 Max|GPU p50 range|GPU p95 range|Engine logical resources|
|---|---:|---:|---:|
|Sponza full F10–F14, dense GI|39.915–42.879 ms|40.665–43.755 ms|11.197 GiB|
|Many Lights1024, RT visibility+custom|24.963–25.470 ms|25.567–25.983 ms|5.893 GiB|
|Forced Apple9/reduced full scene|34.341–36.477 ms|35.114–37.157 ms|10.921 GiB|

All nine processes and O7 GPU-allocation checks pass. These are full-profile
costs, not equal-quality technique comparisons or physical M3 measurements.
The atmospheric profiles use the CLI default12:00 starting clock and fixed
unit exposure; their output is not a calibrated image-quality comparison.
Serial GPU timing includes synchronization in the reported CPU span; do not
infer CPU compute bottlenecks from that field. Engine bytes, device allocation
and process footprint remain separate in the JSON.

**Decision:** do not promote dense DDGI to an automatic/default preset from
these results. It is an explicit quality candidate, not a60-fps full-scene
profile. Existing coarse presets are experimental and do not inherit its
quality acceptance. Measured major costs include aerial perspective and
specular atrous passes. No new optimization is activated here. The frame-budget
exit criteria remain open, along with the listed temporal/native boundaries.

## Regression and stop

The final source build passes nine CTest groups, including the independently
authored AO control tests. GPU regression passes33 F7/F8 checks and36 F9 checks,
the F10 alpha/DRS matrix, F13 lifecycle/negative/admission controls, and the
archive reload pair. Preserve the source/measurement commit for each artifact;
later documentation or runner changes do not relabel older captures.

Stop development at this F14 integration boundary. Keep unfinished qualification
and budget tasks explicit in the roadmap; do not start F15 or automatically
activate the OPT catalog. Future work can use these measured workloads to select
bounded improvements instead of speculative scheduler or kernel infrastructure.
