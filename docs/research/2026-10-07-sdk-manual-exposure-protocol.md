# F13 wide HDR: analytic manual-exposure control

Source-only patch on `f692593093f016ae8f1c59e5ab28040e4f5b030d`. No C++/SDK build or GPU workload executed by this agent; no shared integration source edits. The previous automatic-exposure failure in `build/f13-wide-auto-v1` remains unchanged evidence.

## Public contract, verified before choosing the control

The exact denoised API requires a **1x1 R16Float exposure texture**, not R32Float. Apple describes its value as a multiplier applied to input color. The local Xcode header `MetalFX.framework/Headers/MTLFXTemporalDenoisedScaler.h:338-348` agrees with the [official denoised exposureTexture documentation](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/exposuretexture). The experiment therefore preserves R16Float and explicitly checks the actual texel after quantization.

Apple's [preExposure documentation](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/preexposure) separately says to declare the existing input premultiplication so the scaler can undo it. The [outputTexture page](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/outputtexture) only identifies the output resource and its storage mode; it does not specify the complete output radiometry equation.

The clearest primary explanation is [WWDC25, Go further with Metal 4 games](https://developer.apple.com/videos/play/wwdc2025/211/), exposure discussion around the 2:13 chapter. Exposure describes displayed visibility approximately matching the tonemapper. Apple states that it is a hint and does not change output brightness. Later in that session, the combined denoising API is described as a superset of the upscaler API. Applying this shared explanation to the denoised scaler is a supported inference, not a separate published formula for its pre-exposed output. It is a reason to measure the unchanged output equation before considering any new restoration term.

## Frozen hypothesis and single change

The unit manual exposure is an inappropriate visibility hint for the extreme physical radiance [368640,128,64]. Set manual exposure analytically to E=0.5/368640, mapping the maximum physical component to about 0.5. Auto exposure is **off**. Do not change the physical scene, q, packed color, guides, denoising strength, frame count or output equation.

R16Float conversion is material and must be reported, not treated as additional output tolerance:

- Analytic E = 1.3563368055555556e-6.
- Requested FP32 = 1.3563368383984198e-06.
- Expected R16Float = 23 * 2^-24 = 1.3709068298339844e-6 (bits 0x0017).
- Effective normalized physical red = 0.50537109375.

The patch uses the requested FP32 in the existing pack parameter and lets the R16Float texture conversion occur normally. It checks all four actual exposure readback probes against the exact expected R16 value. Zero from flushing, stale unit exposure, or another value fails the input-channel check before the output is interpreted. Per-frame metadata separately records requested FP32, expected R16, and **actual_provided_manual_exposure_texture_value** with **actual_manual_exposure_readback=true**. Prewarm/summary have no such GPU texel readback and explicitly mark it unavailable; they do not invent an observation.

## Invariants and interpretation fixed before execution

- One case `wide-hdr-manual-exposure`, exactly 96 actual native frames 0..95 at 128x96.
- Physical [368640,128,64], q=1/64, packed color [5760,2,1], manual E above, auto=false, strength=0.
- Existing physical reconstruction remains `SDK/q`; no division or multiplication by E is added.
- Existing `GATES`, `constant_oracle` and `impulse_oracle` are AST-identical to the base. The 1% constant-output gate and every other gate remain unchanged.
- If input readback fails, the control did not deliver its intended exposure; retain that evidence and do not infer SDK output semantics from it.
- If input readback passes and output recovers the unchanged physical target, that supports the normalization hypothesis for this fixture only. It does not promote production policy or prove all scenes.
- If input readback passes and the target gate fails, retain raw SDK and physical PFMs, all channels and unchanged failure status. Inspect them as measurements; do not fit an exposure compensation or waive the target.

## Patch and source checks

Patch `wide-hdr-manual-exposure.patch`, SHA256 `1f13f16152bbd8d0b13c3c837ef40df0f046e341777a8e35b3a1391c01fa83f3`; ten source/test files. `git apply --check` passed. Fourteen Python synthetic tests passed (`python-tests.log`), including exact R16 quantization, flushed/stale input rejection, metadata-versus-readback distinction and mutual exclusion of auto/manual controls. C++ CLI tests were authored but not compiled or executed. A deliberately nonexistent binary was used for the planner-only smoke, which launched no renderer; its manifest is not for GPU use.

Only `MetalfxDenoise::Options.manualExposure` is new in the general adapter, default 1. It fills an existing pack field. Production CLI/configuration never sets it. Descriptor flags and SDK resource format are unchanged from the auto-exposure candidate. The fixture-only boolean CLI rejects non-wide-HDR, missing pre-exposed policy and simultaneous auto-exposure mode.

## Root commands after applying and building

From `/Users/danielsan/.codex/worktrees/f9-core/phosphor`, freeze a fresh destination first:

```sh
python3 tools/metalfx_denoise_fixture_check.py --binary build/lighting/phosphor --out build/f13-wide-manual-v1 --source-root . --gateway native --frames 96 --resolution 128x96 --denoised-fixture-manual-exposure-control
```

Consume the unchanged plan:

```sh
python3 tools/metalfx_denoise_fixture_check.py --binary build/lighting/phosphor --out build/f13-wide-manual-v1 --source-root . --run --manifest build/f13-wide-manual-v1/sdk-frozen-manifest.json
```

The runner preserves process errors, raw measurements and all failed gates even if the normalization hypothesis is wrong. Do not overwrite the unit-exposure or automatic-exposure evidence.
