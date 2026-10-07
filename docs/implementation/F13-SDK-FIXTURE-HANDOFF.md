# F13 genuine SDK fixtures — SOURCE ONLY / NON VERIFIED

This package contains real SDK input/output GPU hooks, readback and independently
written equations. The writer has not compiled, tested, rendered or run either
runner. No fixture result, SDK lifetime acceptance or production output-unit
policy is claimed. Work stops after F14.

`MetalfxDenoiseFixture` accepts the same injected typed request/retire factory as
the production adapter. `PipelineCache` remains unmodified; its optional gateway
patch stays an unapplied review artifact. The tester owns applying that gateway
and enabling `PHOSPHOR_METALFX_DENOISED_GATEWAY_AVAILABLE`. A cast from ordinary
TemporalScaler, a new compiler or its ownership-cycle workaround is never used.

The root owns the explicit CLI fields `denoisedFixture`, `denoisedFixtureOutput`
and `denoisedFixturePreExposed`. The concrete class API is
`Options{scenario,outputDirectory,preExposedPolicy}`, `prepareFrame(Frame)`,
`addToGraph(graph)`, `bindFrame(executor)`, `consume(completedSlot)`, `finish()`,
`ready()`, `version()`, `status()` and `reportJSON()` returning a JSON string.
The root supplies actual Metal frame/view/slot/extents and calls finish after
GPU idle and slot drain. The root installs the generated selected SDK output
before exposure and bypasses ordinary temporal reconstruction only while ready.
Missing SDK, device support, gateway or nonunit output policy gives a reason and
zero native records; finish returns false with `NOT_EXECUTED_NATIVE`.

The fixture writes actual linear HDR channels through compute; Depth32Float is
written by a legitimate raster fragment `[[depth(any)]]` pass requested from the
existing PipelineCache. Reverse-Z depth is `.1/4`, world normals are signed,
roughness is perceptual, motion is current-to-previous input pixels with +Y down,
and the exposure texture readback must equal 1. Each view/frame-ring slot owns
textures, argument tables and Shared readback arrays through GPU completion.
`GPUFXFixtureParams` is 64 B, `GPUFXFixtureSample` is 64 B in the new shared header.

| Kernel | Buffers | Textures |
|---|---|---|
| fx_fixture_generate | params0 | authored color0 normal1 rough2 diffuse3 specular4 motion5 hit6 reactive7 strength8 |
| fx_fixture_depth_fs | params0 fragment stage | depth attachment |
| fx_fixture_readback | params0, actual SDK+physical RGB1, authored samples2, packed samples3 | SDK0 physical1; authored color2 normal3 rough4 motion5 depth6 diffuse7 specular8; actual SDK-packed color9 normal10 rough11 motion12 depth13 diffuse14 specular15 exposure16 hit17 reactive18 strength19 |

Read-only adapter accessors expose current graph refs for actual SDK half output
and actual packed channels. Consumers declare reads, bind imports and preserve
frame/view/slot tags. The fixture copies all output pixels and four authored plus
four actual packed guide probes. It never substitutes input pixels for SDK output.
Records are named `frame-%06llu-view-%u.json`,
with companion `-sdk.pfm` and `-physical.pfm`. Existing files are rejected. The
schema is `phosphor.metalfx-fixture.v1`; records include actual encode identity,
source/binary/manifest SHA values supplied by the runner, shader generation,
actual channels, both output-unit hypotheses and fixed errors. Invalid numbers
are serialized as null and cannot pass. Summary count identity includes encoded,
readback and pack-check records, plus nonzero steady samples.

Constant fixtures use physical RGB(.5,.25,.125). The wide-HDR fixture uses
(368640,128,64), with explicit input scale 1/64, so the maximum packed channel is
5760. Unit exposure prevents a second tone exposure. The experimental restore
is `physical = actualSDK / preExposure`. [Apple's preExposure documentation](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/preexposure)
specifies input division; it does not establish output units. Both possible
output-unit hypotheses are recorded, and default production policy remains
Unverified. Explicit pre-exposed fixtures select an experiment, not acceptance.

The impulse is a seven-pixel square, not an assumed native impulse identity.
Content is repeated under preExposure 1 and 1/64, with a history reset at 48-frame
phase boundaries and at least 8 warm frames. Independent comparisons use observed
normalized gain, relative L1 shape and support disagreement. Energy conservation
and perfect filter identity are not premises. Constants use 1% relative error;
gain/shape/support gates are 2%, support floor is 0.1% of the observed peak. These
values are frozen before execution; the runner never retries or relaxes them.

Channels exercise normal(.6,0,.8), roughness.04/.9, a moving square and -1 input
pixel motion, checking actual packed values and unit exposure. They do not claim
that the unknown native filter must reproduce the noisy input. Lifecycle uses
four views, actual drawable resize and history cuts. After at least8 actual SDK
encodes, the fixture requests one reload through `PipelineCache::reload` with its
current library; the cache retains the borrowed library itself. Passing requires
native encodes across two generations, multiple actual captured extents/views,
factory requests, submitted retirements and resets. Sync-cache configurations
that cannot reload remain failed/unexercised rather than being waived.

`retirements_submitted` does not prove final object destruction. The optional
existing macOS `/usr/bin/leaks --atExit -- <renderer>` wrapper records a genuine
process-at-exit diagnostic when available. Missing tool, missing summary, leaks,
GPU validation errors, timeouts or wrong exit markers fail the requested check.
Even a zero-leak report does not automatically certify every internal SDK
resource lifetime. Summary keeps `final_destruction_verified=false` and phase/
production-policy promotion false; the tester must review full lifetime evidence.

`tools/metalfx_denoise_fixture_check.py` defaults to PLAN ONLY. With
`--gateway expected-missing`, it freezes a real unavailable-gateway case expecting
EXIT 1, nonempty reason and zero native records. With `--gateway native`, it freezes
unit constant, scaled constant/impulse pairs, channels, four-view resize/cut/reload,
wide-HDR scaled and unverified-policy negative cases. `--run` is the only execution
switch; it requires an existing binary and never builds or installs anything.
`--leaks-at-exit` wraps only lifecycle with the existing leaks tool. Actual output
PFM files are read again, independently compared with equations and joined to
the immutable manifest/source/binary/metallib hashes. Per-case timeout and
`MTL_DEBUG_LAYER_WARNING_MODE=nslog` are recorded. No actual runs are included here.

Register new host `metalfx_denoise_fixture.cpp`, new shader `metalfx_fixture.metal`,
shared header dependency `metalfx_denoise_fixture_layout.h`, portable oracle test
and written Python oracle tests. Root has already authored Engine/CLI/CMake
integration; this document reports source contracts, not a successful build.
