# F13 qualification before an OPT decision — 2026-10-08

Owner order: repair/qualify the remaining F13 work before considering the OPT
catalog. No OPT/F15 implementation is activated. Source integration alone is
not full F13 acceptance. The measured 60-fps target is not relaxed.

## Corrected static specular filtering

The old temporal and atrous filters treated the selected secondary object,
path and distance as stable receiver identity. GGX generates a new Monte Carlo
outcome each frame, including valid zero contributions. A static scene could
therefore reject its own samples and retain the binary hit/miss noise.

Stochastic samples now identify their estimator class explicitly. Their history
validity still requires the primary receiver, generation, material, view,
geometry, roughness and source revisions. Random endpoint/path/distance is not
used as a disocclusion test for these samples. Deterministic samples retain the
old strict endpoint guards. Numerical errors and cross-class/source changes
still reject; no radiance clamp or denoise-strength tuning was introduced.
The 112-byte history and 48-byte sample layouts are unchanged.

An exact independent four-draw Bernoulli enumeration establishes mean2 and
variance1 for the averaged estimator, versus raw variance4. A real renderer
fixture places a rough white conductor in a constant infinite environment.
The centre N=V receiver has closed-form reflectance1-log(2), independently of
the GPU's GGX sampling. Physical incoming RGB is[.03,.035,.045]. The frozen
protocol captures128frames at129×97, checks64steady frames and keeps the
original6% p95/15% worst/2% flicker caps after fixed physical normalization.

Positive: p95 error0.0366191, flicker0.00413637, all runtime checks PASS.
Same-binary old-endpoint negative: p95 error0.888951, flicker0.874724; the
physical gate rejects it despite runtime consistency checks passing.
The negative is explicit (`PHOSPHOR_DIAGNOSTIC_SPECULAR_ENDPOINTS=1`), requires
`--debug-lighting`, and never becomes a product preset. Signal capture10 reads
the actual filtered specular output, separately from raw capture6.

Static Sponza with frozen clock: frame31 now reuses230400/230400 valid specular
pixels, compared with110008/230400 in the original trace. DI/GI/AO stay at100%.
The previously reproduced salt-and-pepper defect disappears in the actual
1920×1080 image. This is a static result; moving-camera/complex dynamic GGX
reference qualification remains distinct and is not claimed completed.

## Native SDK alternatives: failures remain visible

At the PR20 checkpoint, no OS installation or private-pointer release had been attempted. The later [native lifetime and budget follow-up](F13-F14-BUDGET-CLOSURE-2026-10-08.md) identifies and retires the leaked POD timing record on one exact verified framework image; the HDR gate remains open. Public-API
create/release reductions tested:

| Configuration | Live wrappers after release | Residual allocations |
|---|---|---|
|Metal4 factory, asynchronous initialization|0|4×640 B|
|Device-only Metal3 factory|0|4×640 B|
|Metal4 factory without optional masks|0|4×640 B|
|Metal4 factory, Float32 color/output|0|1×640 B|

Public weak references prove wrapper deallocation in each case. These results
rule out the tested initialization/factory/mask/format alternatives as fixes;
they do not justify freeing unknown SDK allocations. The original synchronous
and full-lifecycle failures remain preserved. Production native admission
remains disabled; the custom Float32 path remains available.

The original wide-colored-HDR target[368640,128,64] was also rerun with:

- API-defined skip-denoise mask1 on the noise-free fixture;
- SDK-facing color and output bothRGBA32Float, checked against the returned
  scaler formats and actual resource descriptors.

Both execute96native frames and still reach maximum relative error5.17578125%,
outside the unchanged1% gate. Float32 declaration is not treated as radiometric
qualification. The full-strength/half-format failed controls are not replaced.
These are diagnostic-only opt-ins, with explicit metadata and no production
CLI bypass. Native F13.3 remains externally limited on this runtime.

Sources: Apple's [initialization contract](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerdescriptor/requiressynchronousinitialization)
and [mask API](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/denoisestrengthmasktexture).
The API documentation defines1 as ignore-denoise, while a WWDC26 transcript
uses the opposite wording. This experiment follows the property contract and
records the actual bound mask; no polarity change was guessed from the video.

## Validation and cost

Release app/MSL build and nine CTest groups pass. F13 four-view/resize lifecycle,
foreign-history, motion, normal negatives and requested-native/custom fallback
all pass their functional gates. The independent AO control remains green.
The corpus still declares missing broad independent image references explicitly.

Three new full-Sponza1080p runs, same profile/128warmup+256measured frames as
before: GPU p50=40.1316/40.2942/41.9031ms, p95=40.4749/40.8197/42.6124ms,
zero steady GPU allocations. These are unpaired cost observations after a
correctness change, not a claimed speedup. The 60-fps budget remains unmet.
OPT selection is left to the next owner turn, using these real bottlenecks.

Evidence: [stochastic/reflection ledger](../results/F13-stochastic-qualification-M5Max-2026-10-08.json)
and [native public controls](../results/F13-native-public-controls-M5Max-2026-10-08.json).
Raw logs, images, frozen plans and tested public reductions remain under
`build/f13-qualification` in the aggregator worktree. Neither native SDK
qualification nor all F13.6 exit criteria is marked complete by this result.
